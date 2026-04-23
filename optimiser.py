# optimiser.py
# ============================================================
# The Optuna-based Bayesian optimisation engine.
# Ties together validator, simulator, and reward into one
# Optuna study per strategy.
#
# PERFORMANCE: Uses the vectorized NumPy functions from
# simulator.py (replay_combo_np) in the Optuna hot loop.
# All trade dates
# are replayed simultaneously per trial using padded 2D
# NumPy arrays — no Python loops over days or candles.
#
# The original Python simulator (simulator.py) is used ONCE
# at the end to re-replay the winning combo for detailed
# per-day results needed by db_writer.py.
#
# Responsibilities:
#   1. Build the Optuna parameter search space in points
#   2. Preprocess candle data into NumPy arrays (once)
#   3. Define the objective function (one trial = one combo)
#   4. Run the study for N trials or until timeout
#   5. Re-replay winner for detailed results
#   6. Return the winner and full trial record for db_writer
#
# Called by main.py once per strategy.
# Operates entirely in POINTS.
# No INR, no HISTORICAL_CAPITAL inside this file.
# ============================================================

import logging
import time
from typing import Optional

import optuna

from config import (
    OPTUNA_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    UNIVERSAL_EXIT_TIMES,
    VALIDATION_SPLIT,
    MIN_DATES_FOR_SPLIT,
    SL_STEP_PTS,
    TSL_ACTIVATION_STEP_PTS,
    PT_STEP_PTS,
    TSL_GAP_MIN_PCT,
    TSL_GAP_MAX_PCT,
    TSL_GAP_PCT_STEP,
    POINTS_PRECISION,
    StrategyBoundaries,
)
from validator import is_valid_combo, build_combo_label
from simulator import (
    preprocess_days,
    replay_combo_np,
    aggregate_results_np,
    replay_combo,
    PreprocessedDays,
    _time_str_to_minutes,
)
from reward import (
    split_train_validation,
    select_winner,
    compute_reward_score,
    normalise_scores_fast,
    RunningMinMax,
    ComboMetrics,
)

logger = logging.getLogger(__name__)

# suppress Optuna's default per-trial logging —
# we handle our own progress logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PRECOMPUTE EXIT TIME MINUTES
# Convert the 4 universal exit time strings to minutes
# once at module load — never parsed inside the hot loop.
# ============================================================

_EXIT_TIME_MINUTES: dict[str, int] = {
    t: _time_str_to_minutes(t) for t in UNIVERSAL_EXIT_TIMES
}


# ============================================================
# PARAMETER SPACE BUILDER
# Defines the search space boundaries for Optuna.
# All values in points, derived from StrategyBoundaries.
# ============================================================

def _suggest_combo(
    trial:      optuna.Trial,
    boundaries: StrategyBoundaries,
) -> tuple[float, float, float, float, str]:
    """
    Ask Optuna to suggest one combo within the valid search
    space boundaries.

    Parameters:
        SL:           floor to ceiling, step SL_STEP_PTS
        TSL_act:      floor to ceiling (buying/selling), step 2
        TSL_gap_pct:  0.50 to 0.99 (% of activation), step 0.01
        PT:           floor to ceiling (buying/selling), step 2
        Exit time:    categorical from config

    Returns:
        (sl_pts, tsl_activation_pts, tsl_gap_pts,
         pt_pts, universal_exit_time)
    """
    # SL: from floor to ceiling in configured steps
    sl_pts = trial.suggest_float(
        "sl_pts",
        boundaries.sl_floor_pts,
        boundaries.sl_ceiling_pts,
        step=SL_STEP_PTS,
    )

    # TSL activation: floor to ceiling (type-dependent)
    tsl_activation_pts = trial.suggest_float(
        "tsl_activation_pts",
        boundaries.tsl_activation_floor_pts,
        boundaries.tsl_activation_ceiling_pts,
        step=TSL_ACTIVATION_STEP_PTS,
    )

    # TSL gap: sampled as percentage of activation (50–99%)
    # Then converted to absolute points
    tsl_gap_pct = trial.suggest_float(
        "tsl_gap_pct",
        TSL_GAP_MIN_PCT,
        TSL_GAP_MAX_PCT,
        step=TSL_GAP_PCT_STEP,
    )
    tsl_gap_pts = round(
        tsl_gap_pct * tsl_activation_pts, POINTS_PRECISION
    )

    # Profit target: floor to ceiling (type-dependent)
    pt_pts = trial.suggest_float(
        "pt_pts",
        boundaries.pt_floor_pts,
        boundaries.pt_ceiling_pts,
        step=PT_STEP_PTS,
    )

    # Universal exit time: categorical from config list
    universal_exit_time = trial.suggest_categorical(
        "universal_exit_time",
        UNIVERSAL_EXIT_TIMES,
    )

    return (
        sl_pts,
        tsl_activation_pts,
        tsl_gap_pts,
        pt_pts,
        universal_exit_time,
    )


# ============================================================
# OPTUNA OBJECTIVE (VECTORIZED)
# One call = one trial = one combo tested across ALL days
# simultaneously via NumPy.
# Returns reward_score (higher = better).
# ============================================================

def _build_objective(
    train_data:           PreprocessedDays,
    validation_data:      PreprocessedDays,
    per_unit_capital:     float,
    boundaries:           StrategyBoundaries,
    tracker:              RunningMinMax,
    trial_records:        list[dict],
    counters:             dict,
    actual_train_roi_pct: float,
) -> callable:
    """
    Build and return the Optuna objective function.
    Uses closure to capture shared state that persists
    across all trials without global variables.
    """
    has_validation = validation_data.n_days > 0

    def objective(trial: optuna.Trial) -> float:

        counters["tested"] += 1

        # ---- step 1: Optuna suggests a combo ---------------
        (
            sl_pts,
            tsl_activation_pts,
            tsl_gap_pts,
            pt_pts,
            universal_exit_time,
        ) = _suggest_combo(trial, boundaries)

        # ---- step 2: validity gate -------------------------
        if not is_valid_combo(
            sl_pts=sl_pts,
            tsl_activation_pts=tsl_activation_pts,
            tsl_gap_pts=tsl_gap_pts,
            pt_pts=pt_pts,
            universal_exit_time=universal_exit_time,
            boundaries=boundaries,
        ):
            return -1.0

        counters["valid"] += 1

        # precomputed exit time in minutes (from module-level cache)
        univ_minutes = _EXIT_TIME_MINUTES[universal_exit_time]

        # ---- step 3: vectorized replay on train dates ------
        train_pnl, train_win, train_exit = replay_combo_np(
            data=train_data,
            sl_pts=sl_pts,
            tsl_activation_pts=tsl_activation_pts,
            tsl_gap_pts=tsl_gap_pts,
            pt_pts=pt_pts,
            universal_exit_minutes=univ_minutes,
        )

        # ---- step 4: vectorized replay on validation -------
        val_pnl = val_win = val_exit = None
        if has_validation:
            val_pnl, val_win, val_exit = replay_combo_np(
                data=validation_data,
                sl_pts=sl_pts,
                tsl_activation_pts=tsl_activation_pts,
                tsl_gap_pts=tsl_gap_pts,
                pt_pts=pt_pts,
                universal_exit_minutes=univ_minutes,
            )

        # ---- step 5: vectorized aggregation ----------------
        train_metrics = aggregate_results_np(
            train_pnl, train_win, train_exit, per_unit_capital
        )

        validation_metrics = None
        if val_pnl is not None:
            validation_metrics = aggregate_results_np(
                val_pnl, val_win, val_exit, per_unit_capital
            )

        # ---- step 6: O(1) normalise + reward ---------------
        tracker.update(train_metrics)
        normalise_scores_fast(train_metrics, tracker)
        reward = compute_reward_score(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            actual_train_roi_pct=actual_train_roi_pct,
        )

        # ---- step 7: record this trial ---------------------
        trial_records.append({
            "reward_score":        reward,
            "sl_pts":              sl_pts,
            "tsl_activation_pts":  tsl_activation_pts,
            "tsl_gap_pts":         tsl_gap_pts,
            "pt_pts":              pt_pts,
            "universal_exit_time": universal_exit_time,
            "train_metrics":       train_metrics,
            "validation_metrics":  validation_metrics,
        })

        return reward

    return objective


# ============================================================
# PROGRESS LOGGER
# Logs progress every N trials without slowing the loop.
# ============================================================

def _log_progress(
    strategy_id:  int,
    counters:     dict,
    trial_records: list[dict],
    interval:     int = 1000,
) -> None:
    """
    Log optimiser progress every `interval` tested trials.
    Shows tested count, valid count, and current best reward.
    """
    tested = counters["tested"]
    if tested % interval != 0:
        return

    valid   = counters["valid"]
    best    = max(
        (r["reward_score"] for r in trial_records),
        default=0.0,
    )
    pct_valid = round(valid / tested * 100, 1) if tested else 0

    logger.info(
        f"[Strategy {strategy_id}] "
        f"Trials: {tested} tested / {valid} valid "
        f"({pct_valid}%) | "
        f"Best reward so far: {best:.6f}"
    )


# ============================================================
# MAIN OPTIMISER
# Called once per strategy from main.py
# ============================================================

def run_optimiser(
    strategy_id:      int,
    normalised_days:  list[dict],
    per_unit_capital: float,
    boundaries:       StrategyBoundaries,
) -> Optional[dict]:
    """
    Run the full Bayesian optimisation for one strategy.
    """
    n_days = len(normalised_days)
    logger.info(
        f"[Strategy {strategy_id}] Starting optimiser. "
        f"{n_days} total trade dates. "
        f"PER_UNIT_CAPITAL={per_unit_capital}"
    )

    if n_days == 0:
        logger.error(
            f"[Strategy {strategy_id}] "
            f"No normalised days. Cannot optimise."
        )
        return None

    # ---- step 1: walk-forward split ------------------------
    train_days, validation_days = split_train_validation(
        normalised_days=normalised_days,
        validation_split=VALIDATION_SPLIT,
        min_dates=MIN_DATES_FOR_SPLIT,
    )

    # ---- Calculate actual baseline for training set --------
    actual_train_roi_pct = sum(
        day["actual_roi_pct"] for day in train_days 
        if day.get("actual_roi_pct") is not None
    )
    logger.info(
        f"[Strategy {strategy_id}] Baseline Actual ROI "
        f"for training dates: {actual_train_roi_pct}%"
    )

    # ---- step 2: preprocess into NumPy arrays (once) -------
    train_data      = preprocess_days(train_days)
    validation_data = preprocess_days(validation_days)

    logger.info(
        f"[Strategy {strategy_id}] Preprocessed: "
        f"train={train_data.n_days}d × {train_data.max_candles}c | "
        f"val={validation_data.n_days}d × {validation_data.max_candles}c"
    )

    # ---- step 3: shared state across all trials ------------
    tracker:       RunningMinMax    = RunningMinMax()
    trial_records: list[dict]       = []
    counters:      dict             = {
        "tested": 0,
        "valid":  0,
    }

    # ---- step 4: build Optuna study ------------------------
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50,
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    # ---- step 5: build objective with progress callback ----
    objective = _build_objective(
        train_data=train_data,
        validation_data=validation_data,
        per_unit_capital=per_unit_capital,
        boundaries=boundaries,
        tracker=tracker,
        trial_records=trial_records,
        counters=counters,
        actual_train_roi_pct=actual_train_roi_pct,
    )

    def objective_with_logging(trial: optuna.Trial) -> float:
        result = objective(trial)
        _log_progress(
            strategy_id=strategy_id,
            counters=counters,
            trial_records=trial_records,
        )
        return result

    # ---- step 6: run the study -----------------------------
    start_time = time.time()
    logger.info(
        f"[Strategy {strategy_id}] "
        f"Running up to {OPTUNA_TRIALS} trials "
        f"(timeout: {OPTUNA_TIMEOUT_SEC}s)..."
    )

    study.optimize(
        objective_with_logging,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT_SEC,
        show_progress_bar=False,
        n_jobs=1,
    )

    elapsed = round(time.time() - start_time, 1)
    logger.info(
        f"[Strategy {strategy_id}] "
        f"Optimisation complete in {elapsed}s. "
        f"Tested: {counters['tested']} | "
        f"Valid: {counters['valid']} | "
        f"Recorded: {len(trial_records)}"
    )

    # ---- step 7: select winner -----------------------------
    if not trial_records:
        logger.error(
            f"[Strategy {strategy_id}] "
            f"No valid combos found after "
            f"{counters['tested']} trials. "
            f"Check config boundaries — they may be "
            f"too tight for this strategy's capital."
        )
        return None

    winner = select_winner(trial_records)

    if not winner:
        return None

    # ---- step 8: re-replay winner for detailed results -----
    logger.info(
        f"[Strategy {strategy_id}] "
        f"Re-replaying winner for detailed results..."
    )

    winner["train_results"] = replay_combo(
        normalised_days=train_days,
        sl_pts=winner["sl_pts"],
        tsl_activation_pts=winner["tsl_activation_pts"],
        tsl_gap_pts=winner["tsl_gap_pts"],
        pt_pts=winner["pt_pts"],
        universal_exit_time=winner["universal_exit_time"],
    )

    if validation_days:
        winner["validation_results"] = replay_combo(
            normalised_days=validation_days,
            sl_pts=winner["sl_pts"],
            tsl_activation_pts=winner["tsl_activation_pts"],
            tsl_gap_pts=winner["tsl_gap_pts"],
            pt_pts=winner["pt_pts"],
            universal_exit_time=winner["universal_exit_time"],
        )
    else:
        winner["validation_results"] = []

    # ---- step 9: attach metadata for db_writer -------------
    winner["combos_tested"]    = counters["tested"]
    winner["combos_valid"]     = counters["valid"]
    winner["train_dates"]      = len(train_days)
    winner["validation_dates"] = len(validation_days)
    winner["total_dates"]      = n_days

    logger.info(
        f"[Strategy {strategy_id}] Winner: "
        f"SL={winner['sl_pts']}pts | "
        f"TSL_act={winner['tsl_activation_pts']}pts | "
        f"TSL_gap={winner['tsl_gap_pts']}pts | "
        f"PT={winner['pt_pts']}pts | "
        f"Exit={winner['universal_exit_time']} | "
        f"Reward={winner['reward_score']:.6f} | "
        f"TrainROI={winner['train_metrics'].total_roi_pct}% | "
        f"WinRate={winner['train_metrics'].win_rate_pct}%"
    )

    return winner


# ============================================================
# BOUNDARY DIAGNOSTICS
# ============================================================

def log_search_space(
    strategy_id:  int,
    boundaries:   StrategyBoundaries,
    current_lot:  int,
    n_days:       int,
) -> None:
    """
    Log the effective search space for one strategy.
    Called by main.py before run_optimiser().
    """
    def pts_to_inr(pts: float) -> int:
        return round(pts * current_lot)

    kind = "BUYING" if boundaries.is_buying else "SELLING"

    logger.info(
        f"[Strategy {strategy_id}] Search space ({kind}):\n"
        f"  Trade dates:        {n_days}\n"
        f"  PER_UNIT_CAPITAL:   {boundaries.per_unit_capital}\n"
        f"  Current lot size:   {current_lot}\n"
        f"  SL range:           "
        f"{boundaries.sl_floor_pts}–"
        f"{boundaries.sl_ceiling_pts} pts  "
        f"(₹{pts_to_inr(boundaries.sl_floor_pts)}–"
        f"₹{pts_to_inr(boundaries.sl_ceiling_pts)})\n"
        f"  TSL act range:      "
        f"{boundaries.tsl_activation_floor_pts}–"
        f"{boundaries.tsl_activation_ceiling_pts} pts  "
        f"(₹{pts_to_inr(boundaries.tsl_activation_floor_pts)}–"
        f"₹{pts_to_inr(boundaries.tsl_activation_ceiling_pts)})\n"
        f"  TSL gap:            "
        f"{int(TSL_GAP_MIN_PCT*100)}%–"
        f"{int(TSL_GAP_MAX_PCT*100)}% of activation\n"
        f"  PT range:           "
        f"{boundaries.pt_floor_pts}–"
        f"{boundaries.pt_ceiling_pts} pts  "
        f"(₹{pts_to_inr(boundaries.pt_floor_pts)}–"
        f"₹{pts_to_inr(boundaries.pt_ceiling_pts)})\n"
        f"  Universal exits:    {UNIVERSAL_EXIT_TIMES}\n"
        f"  Trials planned:     {OPTUNA_TRIALS}\n"
        f"  Timeout:            {OPTUNA_TIMEOUT_SEC}s"
    )
