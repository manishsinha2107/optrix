# reward.py
# ============================================================
# Computes the reward score for one combo's results.
# Called once per Optuna trial after replay_combo() returns.
#
# Two responsibilities:
#   1. Aggregate daily replay results into summary metrics
#      (total ROI, win rate, max drawdown) for one date set
#   2. Compute the final weighted reward score, implementing 
#      an Alpha Penalty Floor to ensure the engine targets 
#      relative outperformance over historical actuals.
#
# Priority order (from config):
#   ROI 0.70 > Win Rate 0.20 > Drawdown 0.10
#
# All aggregation works in POINTS.
# ROI% uses PER_UNIT_CAPITAL (not HISTORICAL_CAPITAL) —
# this keeps ROI% lot-size agnostic and consistent across
# all trade dates regardless of when they were traded.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import (
    REWARD_WEIGHT_ROI,
    REWARD_WEIGHT_WINRATE,
    REWARD_WEIGHT_DRAWDOWN,
    VALIDATION_PENALTY_THRESHOLD,
    VALIDATION_PENALTY_FACTOR,
    PCT_PRECISION,
    POINTS_PRECISION,
    INR_PRECISION,
)
from simulator import EXIT_SL, EXIT_TSL, EXIT_PT, EXIT_UNIVERSAL

logger = logging.getLogger(__name__)


# ============================================================
# METRICS STRUCTURE
# Aggregated results for one set of trade dates.
# Used for both train set and validation set.
# ============================================================

@dataclass
class ComboMetrics:
    """
    Aggregated performance metrics for one combo
    across one set of trade dates (train or validation).

    All PNL values in points.
    ROI% uses PER_UNIT_CAPITAL as denominator.
    """
    trade_date_count:  int     = 0

    # PNL aggregates (in points)
    total_pnl_pts:     float   = 0.0
    total_roi_pct:     float   = 0.0    # sum of daily roi_pct
    avg_roi_pct:       float   = 0.0    # total_roi_pct / days

    # Win/loss
    winning_trades:    int     = 0
    losing_trades:     int     = 0
    win_rate_pct:      float   = 0.0

    # Drawdown — max single day loss in points
    max_drawdown_pts:  float   = 0.0    # worst single day sim_pnl
    max_drawdown_pct:  float   = 0.0    # as % via PER_UNIT_CAPITAL

    # Exit type breakdown
    sl_exit_count:     int     = 0
    tsl_exit_count:    int     = 0
    pt_exit_count:     int     = 0
    univ_exit_count:   int     = 0

    # Normalised scores (0.0 – 1.0) for reward computation
    roi_score:         float   = 0.0
    winrate_score:     float   = 0.0
    drawdown_score:    float   = 0.0    # lower is better internally
                                        # inverted before weighting


# ============================================================
# 1. AGGREGATE DAILY RESULTS
# ============================================================

def aggregate_results(
    daily_results:    list[dict],
    per_unit_capital: float,
) -> ComboMetrics:
    """
    Aggregate replay_combo() daily results into ComboMetrics.
    """
    if not daily_results:
        return ComboMetrics()

    metrics = ComboMetrics()
    metrics.trade_date_count = len(daily_results)

    total_pnl_pts  = 0.0
    total_roi_pct  = 0.0
    winning        = 0
    worst_day_pts  = 0.0   # most negative single day

    for result in daily_results:
        pnl_pts   = result["sim_pnl_pts"]
        exit_type = result["exit_type"]

        # accumulate PNL
        total_pnl_pts += pnl_pts

        # daily ROI% — lot-size agnostic
        daily_roi_pct = round(
            pnl_pts / per_unit_capital * 100,
            PCT_PRECISION
        )
        total_roi_pct += daily_roi_pct

        # win/loss
        if result["sim_win"]:
            winning += 1

        # drawdown — track worst single day
        if pnl_pts < worst_day_pts:
            worst_day_pts = pnl_pts

        # exit type counts
        if exit_type == EXIT_SL:
            metrics.sl_exit_count += 1
        elif exit_type == EXIT_TSL:
            metrics.tsl_exit_count += 1
        elif exit_type == EXIT_PT:
            metrics.pt_exit_count += 1
        elif exit_type == EXIT_UNIVERSAL:
            metrics.univ_exit_count += 1

    n = metrics.trade_date_count

    metrics.total_pnl_pts    = round(total_pnl_pts, POINTS_PRECISION)
    metrics.total_roi_pct    = round(total_roi_pct, PCT_PRECISION)
    metrics.avg_roi_pct      = round(total_roi_pct / n, PCT_PRECISION)
    metrics.winning_trades   = winning
    metrics.losing_trades    = n - winning
    metrics.win_rate_pct     = round(winning / n * 100, PCT_PRECISION)
    metrics.max_drawdown_pts = round(worst_day_pts, POINTS_PRECISION)
    metrics.max_drawdown_pct = round(
        worst_day_pts / per_unit_capital * 100,
        PCT_PRECISION
    )

    return metrics


# ============================================================
# 2. NORMALISE METRICS TO 0-1 SCORES
# ============================================================

def normalise_scores(
    metrics:          ComboMetrics,
    all_train_metrics: list[ComboMetrics],
) -> ComboMetrics:
    """
    Normalise ROI, win rate, and drawdown to 0.0-1.0 scores
    relative to the range seen across all trials so far.
    """
    if not all_train_metrics or len(all_train_metrics) < 2:
        # not enough data to normalise — assign mid scores
        metrics.roi_score      = 0.5
        metrics.winrate_score  = 0.5
        metrics.drawdown_score = 0.5
        return metrics

    all_roi       = [m.total_roi_pct  for m in all_train_metrics]
    all_winrate   = [m.win_rate_pct   for m in all_train_metrics]
    all_drawdown  = [m.max_drawdown_pct for m in all_train_metrics]

    def minmax(value: float, values: list[float]) -> float:
        lo, hi = min(values), max(values)
        if hi == lo:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    metrics.roi_score     = minmax(metrics.total_roi_pct,   all_roi)
    metrics.winrate_score = minmax(metrics.win_rate_pct,    all_winrate)

    # drawdown: normalise then invert (lower drawdown = higher score)
    raw_dd_score          = minmax(metrics.max_drawdown_pct, all_drawdown)
    metrics.drawdown_score = round(1.0 - raw_dd_score, 4)

    return metrics


# ============================================================
# 2b. RUNNING MIN-MAX TRACKER (O(1) per trial)
# Replaces the O(n) list scan in normalise_scores() for
# the vectorized optimiser loop.
# ============================================================

class RunningMinMax:
    """
    Tracks running min/max for ROI, win rate, and drawdown
    across all completed trials. Updated in O(1) per trial
    instead of scanning the full metrics list.
    """

    def __init__(self):
        self.count:    int   = 0

        self.roi_min:  float = float("inf")
        self.roi_max:  float = float("-inf")

        self.wr_min:   float = float("inf")
        self.wr_max:   float = float("-inf")

        self.dd_min:   float = float("inf")
        self.dd_max:   float = float("-inf")

    def update(self, metrics: ComboMetrics) -> None:
        self.count += 1

        roi = metrics.total_roi_pct
        if roi < self.roi_min:
            self.roi_min = roi
        if roi > self.roi_max:
            self.roi_max = roi

        wr = metrics.win_rate_pct
        if wr < self.wr_min:
            self.wr_min = wr
        if wr > self.wr_max:
            self.wr_max = wr

        dd = metrics.max_drawdown_pct
        if dd < self.dd_min:
            self.dd_min = dd
        if dd > self.dd_max:
            self.dd_max = dd


def normalise_scores_fast(
    metrics: ComboMetrics,
    tracker: RunningMinMax,
) -> ComboMetrics:
    """
    O(1) normalisation using pre-tracked running min/max.
    """
    if tracker.count < 2:
        metrics.roi_score      = 0.5
        metrics.winrate_score  = 0.5
        metrics.drawdown_score = 0.5
        return metrics

    def _minmax(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    metrics.roi_score     = _minmax(
        metrics.total_roi_pct, tracker.roi_min, tracker.roi_max
    )
    metrics.winrate_score = _minmax(
        metrics.win_rate_pct, tracker.wr_min, tracker.wr_max
    )

    raw_dd_score = _minmax(
        metrics.max_drawdown_pct, tracker.dd_min, tracker.dd_max
    )
    metrics.drawdown_score = round(1.0 - raw_dd_score, 4)

    return metrics


# ============================================================
# 3. REWARD SCORE (ALPHA OPTIMISED)
# ============================================================
# Evaluates and scores combinations based on relative 
# outperformance against historical reality.
# ============================================================

def compute_reward_score(
    train_metrics:        ComboMetrics,
    validation_metrics:   Optional[ComboMetrics],
    actual_train_roi_pct: float = 0.0,
) -> float:
    """
    Compute the final weighted reward score for one combo.
    Optimised for Relative Outperformance (Alpha).

    Alpha Penalty Floor:
        If the simulated ROI is worse than the actual historical ROI,
        the ROI component of the reward is hard-capped to 0.0.
        This forces Optuna to hunt exclusively for combos that beat reality.
    """
    # Calculate excess return (Alpha)
    excess_roi = train_metrics.total_roi_pct - actual_train_roi_pct

    # Base weighted components
    roi_component      = train_metrics.roi_score * REWARD_WEIGHT_ROI
    winrate_component  = train_metrics.winrate_score * REWARD_WEIGHT_WINRATE
    drawdown_component = train_metrics.drawdown_score * REWARD_WEIGHT_DRAWDOWN

    # ---- The Alpha Penalty Floor ----
    if excess_roi < 0:
        # If it doesn't beat reality, zero out its ROI score.
        # It can still get a marginal score for Win Rate / Drawdown, 
        # but it will immediately drop to the bottom of Optuna's rankings.
        roi_component = 0.0

    reward = roi_component + winrate_component + drawdown_component

    # validation penalty (applies only if VALIDATION_SPLIT > 0.0)
    if validation_metrics and validation_metrics.trade_date_count > 0:
        roi_gap = (
            train_metrics.avg_roi_pct
            - validation_metrics.avg_roi_pct
        )
        if roi_gap > VALIDATION_PENALTY_THRESHOLD:
            logger.debug(
                f"Validation penalty applied: "
                f"train_avg_roi={train_metrics.avg_roi_pct}% "
                f"val_avg_roi={validation_metrics.avg_roi_pct}% "
                f"gap={roi_gap:.2f}% > "
                f"threshold={VALIDATION_PENALTY_THRESHOLD}%"
            )
            reward *= VALIDATION_PENALTY_FACTOR

    return round(reward, 6)


# ============================================================
# 4. WALK-FORWARD SPLIT
# Splits normalised days into train and validation sets.
# Oldest dates → training. Most recent dates → validation.
# ============================================================

def split_train_validation(
    normalised_days:  list[dict],
    validation_split: float,
    min_dates:        int,
) -> tuple[list[dict], list[dict]]:
    """
    Split normalised_days into train and validation sets.
    """
    n = len(normalised_days)

    if n < min_dates or validation_split == 0.0:
        logger.info(
            f"Walk-forward split disabled or dates < {min_dates}. "
            f"Running on 100% of data."
        )
        return normalised_days, []

    split_idx = int(n * (1 - validation_split))
    # ensure at least 1 date in each set
    split_idx = max(1, min(split_idx, n - 1))

    train_days      = normalised_days[:split_idx]
    validation_days = normalised_days[split_idx:]

    logger.info(
        f"Walk-forward split: {len(train_days)} train dates, "
        f"{len(validation_days)} validation dates "
        f"(most recent {len(validation_days)} held out)."
    )
    return train_days, validation_days


# ============================================================
# 5. FULL SCORING PIPELINE
# Called once per Optuna trial.
# Combines aggregate + normalise + reward in one call.
# ============================================================

def score_combo(
    train_results:        list[dict],
    validation_results:   list[dict],
    per_unit_capital:     float,
    all_train_metrics:    list[ComboMetrics],
    actual_train_roi_pct: float = 0.0,
) -> tuple[float, ComboMetrics, Optional[ComboMetrics]]:
    """
    Full scoring pipeline for one Optuna trial.
    """
    # step 1 — aggregate
    train_metrics = aggregate_results(
        train_results, per_unit_capital
    )

    validation_metrics = None
    if validation_results:
        validation_metrics = aggregate_results(
            validation_results, per_unit_capital
        )

    # step 2 — normalise (appending current to history
    #           is done by the caller in optimiser.py)
    normalise_scores(train_metrics, all_train_metrics)

    # step 3 — reward
    reward = compute_reward_score(
        train_metrics, validation_metrics, actual_train_roi_pct
    )

    return reward, train_metrics, validation_metrics


# ============================================================
# 6. WINNER SELECTOR
# After all Optuna trials complete, pick the best combo.
# ============================================================

def select_winner(
    trial_scores: list[dict],
) -> dict:
    """
    Select the winning combo from all completed trials.
    """
    if not trial_scores:
        logger.error("No valid trials to select winner from.")
        return {}

    winner = max(trial_scores, key=lambda t: t["reward_score"])

    logger.info(
        f"Winner selected: "
        f"reward={winner['reward_score']:.6f} | "
        f"sl={winner['sl_pts']} | "
        f"tsl_act={winner['tsl_activation_pts']} | "
        f"tsl_gap={winner['tsl_gap_pts']} | "
        f"pt={winner['pt_pts']} | "
        f"exit={winner['universal_exit_time']} | "
        f"train_roi={winner['train_metrics'].total_roi_pct}% | "
        f"winrate={winner['train_metrics'].win_rate_pct}%"
    )
    return winner
