# db_writer.py
# ============================================================
# Writes the winning combo and its daily detail to Supabase.
# This is the ONLY file that writes to the database.
#
# Three responsibilities:
#   1. Write one row to sim_combos (the winning combo summary)
#   2. Write N rows to sim_daily (one per trade date)
#   3. Extract and calculate the actual baseline PNL 
#      and ROI% for the specific dates Optuna trained on, 
#      allowing a true 1:1 comparison against the simulation.
#
# All INR conversions happen here — this is the only place
# in the entire engine where points are multiplied by lot
# sizes to produce rupee values for storage and display.
#
# Capital vocabulary in this file:
#   PER_UNIT_CAPITAL   — used for ROI% (already in winner)
#   HISTORICAL_CAPITAL — computed per trade date here
#                        for sim_daily storage only
#   sim_pnl_inr        — sim_pnl_pts × lot_size_on_date
#   combo INR labels   — pts × current_lot_size
#
# Uses upsert (not insert) so re-running the optimiser on
# the same day for the same strategy cleanly replaces the
# previous result without duplicate rows.
# ============================================================

import logging
from datetime import date
from typing import Optional

from supabase import Client

from config import (
    INR_PRECISION,
    PCT_PRECISION,
    POINTS_PRECISION,
)
from validator import build_combo_label
from normaliser import (
    compute_historical_capital,
    pts_to_inr,
)
from reward import ComboMetrics
from simulator import EXIT_SL, EXIT_TSL, EXIT_PT, EXIT_UNIVERSAL

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — SAFE ROUND
# Avoids storing float artifacts like 0.30000000000000004
# ============================================================

def _r(value: float, decimals: int) -> float:
    """Round float to decimals. Returns 0.0 for None."""
    if value is None:
        return None
    return round(float(value), decimals)


# ============================================================
# 1. BUILD sim_combos ROW
# ============================================================
# Constructs the master summary record for the winning combo.
# Now includes total_actual_pnl_inr and total_actual_roi_pct
# to serve as a baseline reality check against the simulation.
# ============================================================

def _build_sim_combos_row(
    strategy_id:          int,
    winner:               dict,
    train_metrics:        ComboMetrics,
    validation_metrics:   Optional[ComboMetrics],
    current_lot_size:     int,
    per_unit_capital:     float,
    run_date:             str,
    total_actual_pnl_inr: float,
    total_actual_roi_pct: float,
) -> dict:
    """
    Build the dict to upsert into sim_combos.

    Converts all point values to INR using current_lot_size.
    This is the display-ready combo for the UI.

    Args:
        strategy_id:          the strategy being optimised
        winner:               output of optimiser.select_winner()
        train_metrics:        ComboMetrics for training dates
        validation_metrics:   ComboMetrics for validation dates
                              or None
        current_lot_size:     current lot size for INR labels
        per_unit_capital:     the invariant
        run_date:             "YYYY-MM-DD" string for today
        total_actual_pnl_inr: sum of actual PNL for training dates
        total_actual_roi_pct: sum of actual ROI% for training dates

    Returns:
        dict ready for supabase upsert into sim_combos
    """
    sl_pts             = winner["sl_pts"]
    tsl_activation_pts = winner["tsl_activation_pts"]
    tsl_gap_pts        = winner["tsl_gap_pts"]
    tsl_floor_pts      = round(
        tsl_activation_pts - tsl_gap_pts,
        POINTS_PRECISION
    )
    pt_pts             = winner["pt_pts"]
    exit_time          = winner["universal_exit_time"]

    # convert winning combo pts → INR at current lot size
    # these are the values shown in the UI combo label
    sl_inr             = _r(
        pts_to_inr(sl_pts, current_lot_size),
        INR_PRECISION
    )
    tsl_activation_inr = _r(
        pts_to_inr(tsl_activation_pts, current_lot_size),
        INR_PRECISION
    )
    tsl_floor_inr      = _r(
        pts_to_inr(tsl_floor_pts, current_lot_size),
        INR_PRECISION
    )
    pt_inr             = _r(
        pts_to_inr(pt_pts, current_lot_size),
        INR_PRECISION
    )

    combo_label = build_combo_label(
        sl_pts=sl_pts,
        tsl_activation_pts=tsl_activation_pts,
        tsl_gap_pts=tsl_gap_pts,
        pt_pts=pt_pts,
        universal_exit_time=exit_time,
        current_lot_size=current_lot_size,
    )

    # validation scores — None if no split was applied
    val_roi    = None
    val_wr     = None
    if validation_metrics and validation_metrics.trade_date_count > 0:
        val_roi = _r(validation_metrics.total_roi_pct, PCT_PRECISION)
        val_wr  = _r(validation_metrics.win_rate_pct,  PCT_PRECISION)

    # max drawdown in INR at current lot size
    max_dd_inr = _r(
        pts_to_inr(
            abs(train_metrics.max_drawdown_pts),
            current_lot_size
        ),
        INR_PRECISION
    )

    return {
        "strategy_id":          strategy_id,
        "run_date":             run_date,

        # combo in points
        "sl_pts":               _r(sl_pts,              POINTS_PRECISION),
        "tsl_activation_pts":   _r(tsl_activation_pts,  POINTS_PRECISION),
        "tsl_gap_pts":          _r(tsl_gap_pts,         POINTS_PRECISION),
        "tsl_floor_pts":        _r(tsl_floor_pts,       POINTS_PRECISION),
        "pt_pts":               _r(pt_pts,              POINTS_PRECISION),
        "universal_exit_time":  exit_time,

        # combo in INR (at current lot size, for UI display)
        "sl_inr":               sl_inr,
        "tsl_activation_inr":   tsl_activation_inr,
        "tsl_floor_inr":        tsl_floor_inr,
        "pt_inr":               pt_inr,

        # human readable UI label
        "combo_label":          combo_label,

        # lot size context
        "current_lot_size":     current_lot_size,
        "per_unit_capital":     _r(per_unit_capital, INR_PRECISION),

        # transparency
        "combos_tested":        winner["combos_tested"],
        "combos_valid":         winner["combos_valid"],
        "trade_dates_used":     winner["total_dates"],
        "train_dates":          winner["train_dates"],
        "validation_dates":     winner["validation_dates"],

        # aggregate scores — training set
        "total_pnl_inr":        _r(
            pts_to_inr(
                train_metrics.total_pnl_pts,
                current_lot_size
            ),
            INR_PRECISION
        ),
        "total_roi_pct":        _r(train_metrics.total_roi_pct,  PCT_PRECISION),
        "win_rate_pct":         _r(train_metrics.win_rate_pct,   PCT_PRECISION),
        "max_drawdown_inr":     max_dd_inr,
        "max_drawdown_pct":     _r(train_metrics.max_drawdown_pct, PCT_PRECISION),

        # actual baseline comparison for training dates
        "total_actual_pnl_inr": _r(total_actual_pnl_inr, INR_PRECISION),
        "total_actual_roi_pct": _r(total_actual_roi_pct, PCT_PRECISION),

        # exit type breakdown
        "sl_exit_count":        train_metrics.sl_exit_count,
        "tsl_exit_count":       train_metrics.tsl_exit_count,
        "pt_exit_count":        train_metrics.pt_exit_count,
        "univ_exit_count":      train_metrics.univ_exit_count,

        # walk-forward validation scores
        "validation_roi_pct":   val_roi,
        "validation_win_rate":  val_wr,

        # final reward score
        "reward_score":         _r(winner["reward_score"], 6),
    }


# ============================================================
# 2. BUILD sim_daily ROWS
# ============================================================
# Constructs the granular date-by-date records.
# This merges the simulated results with the historical reality 
# (actual_pnl, lot_size_on_date, historical_capital) for storage.
# ============================================================

# ============================================================
# 2. BUILD sim_daily ROWS
# ============================================================

def _build_sim_daily_rows(
    strategy_id:       int,
    run_date:          str,
    winner:            dict,
    normalised_days:   list[dict],
    per_unit_capital:  float,
) -> list[dict]:
    """
    Build one sim_daily row per trade date for the winning
    combo. Merges the replay results with the normalised
    day context (lot_size_on_date, historical_capital,
    actual_pnl, la_pnl, plus intraday peak/trough metadata).

    This is where HISTORICAL_CAPITAL is computed per date —
    the only place in the engine it is needed for storage.
    """
    # merge train and validation results into one lookup
    # keyed by trade_date for fast access
    all_results: dict[str, dict] = {}

    for result in winner.get("train_results", []):
        all_results[result["trade_date"]] = result

    for result in winner.get("validation_results", []):
        all_results[result["trade_date"]] = result

    rows = []

    for day in normalised_days:
        trade_date       = day["trade_date"]
        lot_size_on_date = day["lot_size_on_date"]
        
        # Base Actual and LA metrics
        actual_pnl_inr   = day.get("actual_pnl_inr")
        actual_roi_pct   = day.get("actual_roi_pct")
        la_pnl_inr       = day.get("la_pnl_inr")
        la_roi_pct       = day.get("la_roi_pct")

        # HISTORICAL_CAPITAL — computed here for storage only
        historical_capital = compute_historical_capital(
            per_unit_capital=per_unit_capital,
            lot_size_on_date=lot_size_on_date,
        )

        result = all_results.get(trade_date)
        if not result:
            logger.warning(
                f"No replay result for trade_date="
                f"{trade_date}. Skipping sim_daily row."
            )
            continue

        sim_pnl_pts = result["sim_pnl_pts"]

        # sim_pnl_inr — historical rupee value for this date
        sim_pnl_inr = _r(
            pts_to_inr(sim_pnl_pts, lot_size_on_date),
            INR_PRECISION
        )

        # sim_roi_pct — lot-size agnostic via per_unit_capital
        sim_roi_pct = _r(
            sim_pnl_pts / per_unit_capital * 100,
            PCT_PRECISION
        )

        # peak and trough in INR for H @ time / L @ time UI
        peak_pnl_inr = _r(
            pts_to_inr(result["peak_pnl_pts"], lot_size_on_date),
            INR_PRECISION
        )
        trough_pnl_inr = _r(
            pts_to_inr(result["trough_pnl_pts"], lot_size_on_date),
            INR_PRECISION
        )

        rows.append({
            "strategy_id":        strategy_id,
            "run_date":           run_date,
            "trade_date":         trade_date,

            # lot size context for this date
            "lot_size_on_date":   lot_size_on_date,
            "historical_capital": _r(historical_capital, INR_PRECISION),

            # actual EOD pnl (last candle c) and detailed metadata
            "actual_pnl_inr":     _r(actual_pnl_inr, INR_PRECISION),
            "actual_roi_pct":     _r(actual_roi_pct,  PCT_PRECISION),
            "actual_peak_inr":    _r(day.get("actual_peak_inr"), INR_PRECISION),
            "actual_peak_time":   day.get("actual_peak_time"),
            "actual_trough_inr":  _r(day.get("actual_trough_inr"), INR_PRECISION),
            "actual_trough_time": day.get("actual_trough_time"),
            "actual_exit_time":   day.get("actual_exit_time"),

            # live auto pnl (null if no la_mapping_id) and detailed metadata
            "la_pnl_inr":         _r(la_pnl_inr, INR_PRECISION),
            "la_roi_pct":         _r(la_roi_pct,  PCT_PRECISION),
            "la_peak_inr":        _r(day.get("la_peak_inr"), INR_PRECISION),
            "la_peak_time":       day.get("la_peak_time"),
            "la_trough_inr":      _r(day.get("la_trough_inr"), INR_PRECISION),
            "la_trough_time":     day.get("la_trough_time"),
            "la_exit_time":       day.get("la_exit_time"),

            # simulated result for winning combo
            "sim_pnl_pts":        _r(sim_pnl_pts,  POINTS_PRECISION),
            "sim_pnl_inr":        sim_pnl_inr,
            "sim_roi_pct":        sim_roi_pct,
            "sim_win":            result["sim_win"],

            # exit detail
            "exit_type":          result["exit_type"],
            "exit_time":          result["exit_time"],

            # intraday peak and trough for UI H @ / L @ display
            "peak_pnl_inr":       peak_pnl_inr,
            "peak_time":          result["peak_time"],
            "trough_pnl_inr":     trough_pnl_inr,
            "trough_time":        result["trough_time"],
        })

    return rows


# ============================================================
# 3. UPSERT sim_combos
# ============================================================

def write_sim_combos(
    client:       Client,
    row:          dict,
) -> bool:
    """
    Upsert one row into sim_combos.
    Uses upsert on (strategy_id, run_date) unique constraint.
    """
    try:
        client.table("sim_combos").upsert(
            row,
            on_conflict="strategy_id,run_date",
        ).execute()

        logger.info(
            f"sim_combos written: "
            f"strategy_id={row['strategy_id']} | "
            f"run_date={row['run_date']} | "
            f"combo={row['combo_label']}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to write sim_combos for "
            f"strategy_id={row['strategy_id']}: {e}"
        )
        return False


# ============================================================
# 4. UPSERT sim_daily
# ============================================================

def write_sim_daily(
    client: Client,
    rows:   list[dict],
) -> bool:
    """
    Upsert all sim_daily rows for one strategy's winning combo.
    Batches into chunks of 500 to stay within Supabase limits.
    """
    if not rows:
        logger.warning("write_sim_daily called with empty rows.")
        return True

    strategy_id = rows[0]["strategy_id"]
    batch_size  = 500
    success     = True

    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        try:
            client.table("sim_daily").upsert(
                batch,
                on_conflict="strategy_id,run_date,trade_date",
            ).execute()

            logger.info(
                f"sim_daily batch written: "
                f"strategy_id={strategy_id} | "
                f"rows {i + 1}–{i + len(batch)} "
                f"of {len(rows)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to write sim_daily batch "
                f"{i}–{i + len(batch)} for "
                f"strategy_id={strategy_id}: {e}"
            )
            success = False

    return success


# ============================================================
# 5. FULL WRITE PIPELINE
# ============================================================
# Called once per strategy from main.py after optimiser completes. 
# Calculates the baseline actuals using ONLY the dates Optuna 
# used for training, ensuring a strict apples-to-apples comparison.
# Writes both tables in the correct order.
# ============================================================

def write_results(
    client:           Client,
    strategy_id:      int,
    winner:           dict,
    normalised_days:  list[dict],
    current_lot_size: int,
    per_unit_capital: float,
    run_date:         str,
) -> bool:
    """
    Full write pipeline for one strategy's winning combo.
    Writes sim_combos first, then sim_daily.
    """
    train_metrics      = winner["train_metrics"]
    validation_metrics = winner.get("validation_metrics")

    # ---- compute actual baseline for training dates ----
    # We must only sum the actual PNL for the dates Optuna 
    # actually trained against. If we included validation dates 
    # here, the baseline would be skewed against train_metrics.
    train_dates_set = {
        res["trade_date"] for res in winner.get("train_results", [])
    }
    
    train_actuals = [
        day for day in normalised_days
        if day["trade_date"] in train_dates_set and day.get("actual_pnl_inr") is not None
    ]
    
    total_actual_pnl_inr = sum(
        day["actual_pnl_inr"] for day in train_actuals if day.get("actual_pnl_inr") is not None
    )
    total_actual_roi_pct = sum(
        day["actual_roi_pct"] for day in train_actuals if day.get("actual_roi_pct") is not None
    )
    # ---------------------------------------------------------

    # ---- build sim_combos row ------------------------------
    combos_row = _build_sim_combos_row(
        strategy_id=strategy_id,
        winner=winner,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        current_lot_size=current_lot_size,
        per_unit_capital=per_unit_capital,
        run_date=run_date,
        total_actual_pnl_inr=total_actual_pnl_inr,
        total_actual_roi_pct=total_actual_roi_pct,
    )

    # ---- build sim_daily rows ------------------------------
    daily_rows = _build_sim_daily_rows(
        strategy_id=strategy_id,
        run_date=run_date,
        winner=winner,
        normalised_days=normalised_days,
        per_unit_capital=per_unit_capital,
    )

    logger.info(
        f"[Strategy {strategy_id}] Writing results: "
        f"1 sim_combos row + {len(daily_rows)} sim_daily rows."
    )

    # ---- write sim_combos ----------------------------------
    combos_ok = write_sim_combos(client, combos_row)
    if not combos_ok:
        logger.error(
            f"[Strategy {strategy_id}] "
            f"sim_combos write failed. "
            f"Aborting sim_daily write to avoid orphaned rows."
        )
        return False

    # ---- write sim_daily -----------------------------------
    daily_ok = write_sim_daily(client, daily_rows)
    if not daily_ok:
        logger.error(
            f"[Strategy {strategy_id}] "
            f"sim_daily write partially failed. "
            f"sim_combos row was written — manual cleanup "
            f"may be needed for trade_date gaps."
        )
        return False

    logger.info(
        f"[Strategy {strategy_id}] "
        f"All results written successfully."
    )
    return True
