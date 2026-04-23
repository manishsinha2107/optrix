# simulator.py
# ============================================================
# The trade replay engine — both vectorized and sequential.
#
# This file contains TWO replay implementations:
#
#   1. VECTORIZED (NumPy) — replay_combo_np()
#      Replays one combo across ALL trade dates simultaneously
#      using padded 2D NumPy arrays. Called 50,000× per
#      strategy from the Optuna hot loop. No Python loops.
#
#   2. SEQUENTIAL (Python) — replay_trade() / replay_combo()
#      Replays one combo against one trade date at a time.
#      Used ONCE after optimisation to re-replay the winning
#      combo for detailed per-day results (exit times, peak
#      times, etc.) needed by db_writer.py.
#
# Exit priority order (same in both implementations):
#   PT > SL > TSL > UNIVERSAL
#
# All values in POINTS throughout.
# INR conversion happens in db_writer.py at output stage.
# ============================================================

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from config import POINTS_PRECISION, PCT_PRECISION

logger = logging.getLogger(__name__)


# ============================================================
# EXIT TYPE CONSTANTS (string labels)
# Used by db_writer.py and the sequential replay.
# ============================================================

EXIT_SL        = "SL"
EXIT_TSL       = "TSL"
EXIT_PT        = "PT"
EXIT_UNIVERSAL = "UNIVERSAL"


# ============================================================
# EXIT TYPE INDEX CONSTANTS (for vectorized replay)
# PT has index 0 so it wins ties in np.argmin.
# ============================================================

EXIT_PT_IDX:   int = 0
EXIT_SL_IDX:   int = 1
EXIT_TSL_IDX:  int = 2
EXIT_UNIV_IDX: int = 3

EXIT_TYPE_LABELS = {
    EXIT_PT_IDX:   "PT",
    EXIT_SL_IDX:   "SL",
    EXIT_TSL_IDX:  "TSL",
    EXIT_UNIV_IDX: "UNIVERSAL",
}


# ============================================================
# TIME PARSER
# Converts "10:15 AM" / "3:10 PM" / "14:30" to minutes
# since midnight as an integer.
# Used by both preprocessing (once) and sequential replay.
# ============================================================

def _time_str_to_minutes(t: str) -> int:
    """
    Convert a time string to minutes since midnight.

    Handles two formats:
        "10:15 AM" / "3:10 PM"  — 12h from pnl_data
        "14:30"                 — 24h from config

    Returns:
        Integer minutes since midnight.
        e.g. "2:30 PM" → 870, "14:30" → 870
    """
    t = t.strip()
    if "AM" in t or "PM" in t:
        is_pm = "PM" in t
        parts = t.replace("AM", "").replace("PM", "").strip()
        h_str, m_str = parts.split(":")
        h, m = int(h_str), int(m_str)
        if is_pm and h != 12:
            h += 12
        elif not is_pm and h == 12:
            h = 0
        return h * 60 + m
    else:
        h_str, m_str = t.split(":")
        return int(h_str) * 60 + int(m_str)


# ############################################################
#
#  PART 1: VECTORIZED NUMPY REPLAY
#  Used in the Optuna hot loop (50,000× per strategy)
#
# ############################################################


# ============================================================
# PREPROCESSED DATA STRUCTURE
# Created once per strategy before the Optuna loop.
# ============================================================

@dataclass
class PreprocessedDays:
    """
    Candle data for N trade dates packed into padded
    2D NumPy arrays of shape (n_days, max_candles).

    Padding values:
        h → -inf   (never triggers h >= pt_pts)
        l → +inf   (never triggers l <= -sl_pts)
        c → 0.0    (safe for UNIVERSAL exit PNL)
        time_min → 9999  (never triggers time >= exit_time)
    """
    h:           np.ndarray   # (n_days, max_candles) float64
    l:           np.ndarray   # (n_days, max_candles) float64
    c:           np.ndarray   # (n_days, max_candles) float64
    time_min:    np.ndarray   # (n_days, max_candles) int32
    n_candles:   np.ndarray   # (n_days,) int32
    n_days:      int
    max_candles: int
    trade_dates: list


def preprocess_days(normalised_days: list[dict]) -> PreprocessedDays:
    """
    Convert normalised day dicts into padded NumPy arrays.

    Called once per date set (train set, validation set)
    before the Optuna loop begins. This is O(total_candles)
    and runs in ~1ms for 100 days × 350 candles.

    Args:
        normalised_days: output of normaliser.normalise_all_days()
                         or a subset (train/validation split)

    Returns:
        PreprocessedDays ready for replay_combo_np()
    """
    n_days = len(normalised_days)

    if n_days == 0:
        return PreprocessedDays(
            h=np.empty((0, 0), dtype=np.float64),
            l=np.empty((0, 0), dtype=np.float64),
            c=np.empty((0, 0), dtype=np.float64),
            time_min=np.empty((0, 0), dtype=np.int32),
            n_candles=np.empty(0, dtype=np.int32),
            n_days=0,
            max_candles=0,
            trade_dates=[],
        )

    candle_counts = [len(d["candles_pts"]) for d in normalised_days]
    max_candles = max(candle_counts)

    h_arr      = np.full((n_days, max_candles), -np.inf, dtype=np.float64)
    l_arr      = np.full((n_days, max_candles),  np.inf, dtype=np.float64)
    c_arr      = np.zeros((n_days, max_candles),         dtype=np.float64)
    time_arr   = np.full((n_days, max_candles), 9999,    dtype=np.int32)
    n_candles  = np.array(candle_counts,                 dtype=np.int32)

    trade_dates = []

    for i, day in enumerate(normalised_days):
        candles = day["candles_pts"]
        n = len(candles)
        trade_dates.append(day["trade_date"])

        for j in range(n):
            candle = candles[j]
            h_arr[i, j]    = candle["h_pts"]
            l_arr[i, j]    = candle["l_pts"]
            c_arr[i, j]    = candle["c_pts"]
            time_arr[i, j] = _time_str_to_minutes(candle["time"])

    return PreprocessedDays(
        h=h_arr,
        l=l_arr,
        c=c_arr,
        time_min=time_arr,
        n_candles=n_candles,
        n_days=n_days,
        max_candles=max_candles,
        trade_dates=trade_dates,
    )


# ============================================================
# VECTORIZED REPLAY
# The core hot-path function. Called 50,000× per strategy.
# All operations fully vectorized — no Python loops.
# ============================================================

def replay_combo_np(
    data:                  PreprocessedDays,
    sl_pts:                float,
    tsl_activation_pts:    float,
    tsl_gap_pts:           float,
    pt_pts:                float,
    universal_exit_minutes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized replay of one combo across all trade dates.

    Args:
        data:                   PreprocessedDays from preprocess_days()
        sl_pts:                 stop loss in points (positive)
        tsl_activation_pts:     TSL activation level in points
        tsl_gap_pts:            TSL gap in points
        pt_pts:                 profit target in points
        universal_exit_minutes: exit time as minutes since midnight

    Returns:
        sim_pnl:    (n_days,) float64 — PNL in points per day
        sim_win:    (n_days,) bool    — whether each day was a win
        exit_type:  (n_days,) int32   — exit type index
                    0=PT, 1=SL, 2=TSL, 3=UNIVERSAL
    """
    if data.n_days == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.int32),
        )

    n_days  = data.n_days
    max_c   = data.max_candles
    sl_neg  = -sl_pts

    # cumulative peak (running max of h)
    peaks = np.maximum.accumulate(data.h, axis=1)

    # candle index array for masking
    candle_idx = np.arange(max_c, dtype=np.int32)[np.newaxis, :]

    # PROFIT TARGET: first candle where h >= pt_pts
    pt_hit = data.h >= pt_pts
    pt_any = pt_hit.any(axis=1)
    pt_idx = np.where(pt_any, np.argmax(pt_hit, axis=1), max_c)

    # STOP LOSS: first candle where l <= -sl_pts
    sl_hit = data.l <= sl_neg
    sl_any = sl_hit.any(axis=1)
    sl_idx = np.where(sl_any, np.argmax(sl_hit, axis=1), max_c)

    # TSL ARMING: first candle where peak >= activation
    tsl_arm_hit  = peaks >= tsl_activation_pts
    tsl_armed_any = tsl_arm_hit.any(axis=1)
    tsl_arm_idx  = np.where(
        tsl_armed_any,
        np.argmax(tsl_arm_hit, axis=1),
        max_c,
    )

    # TSL BREACH: l <= (peak - gap) AND candle >= arm index
    tsl_floor   = peaks - tsl_gap_pts
    tsl_breach  = (data.l <= tsl_floor) & (candle_idx >= tsl_arm_idx[:, np.newaxis])
    tsl_breach_any = tsl_breach.any(axis=1)
    tsl_idx     = np.where(
        tsl_breach_any,
        np.argmax(tsl_breach, axis=1),
        max_c,
    )

    # UNIVERSAL EXIT: first candle where time >= exit minutes
    univ_hit = data.time_min >= universal_exit_minutes
    univ_any = univ_hit.any(axis=1)
    last_candle_idx = np.maximum(data.n_candles - 1, 0)
    univ_idx = np.where(
        univ_any,
        np.argmax(univ_hit, axis=1),
        last_candle_idx,
    )

    # RESOLVE: earliest candle wins, ties → PT priority
    exit_indices = np.stack(
        [pt_idx, sl_idx, tsl_idx, univ_idx], axis=1
    )
    exit_type    = np.argmin(exit_indices, axis=1).astype(np.int32)
    exit_at      = exit_indices[np.arange(n_days), exit_type]

    # COMPUTE PNL per day
    sim_pnl = np.empty(n_days, dtype=np.float64)

    is_pt   = exit_type == EXIT_PT_IDX
    is_sl   = exit_type == EXIT_SL_IDX
    is_tsl  = exit_type == EXIT_TSL_IDX
    is_univ = exit_type == EXIT_UNIV_IDX

    sim_pnl[is_pt] = pt_pts
    sim_pnl[is_sl] = sl_neg

    if np.any(is_tsl):
        tsl_days = np.where(is_tsl)[0]
        sim_pnl[is_tsl] = peaks[tsl_days, exit_at[is_tsl]] - tsl_gap_pts

    if np.any(is_univ):
        univ_days = np.where(is_univ)[0]
        sim_pnl[is_univ] = data.c[univ_days, exit_at[is_univ]]

    sim_win = sim_pnl > 0

    return sim_pnl, sim_win, exit_type


# ============================================================
# VECTORIZED AGGREGATION
# Replaces the Python loop in reward.aggregate_results().
# ============================================================

def aggregate_results_np(
    sim_pnl:          np.ndarray,
    sim_win:          np.ndarray,
    exit_type:        np.ndarray,
    per_unit_capital: float,
) -> "ComboMetrics":
    """
    Aggregate vectorized replay results into ComboMetrics.

    Produces identical ComboMetrics output to
    reward.aggregate_results() but from NumPy arrays
    instead of a list of dicts.

    Args:
        sim_pnl:          (n_days,) float64 — PNL per day
        sim_win:          (n_days,) bool    — win per day
        exit_type:        (n_days,) int32   — exit type index
        per_unit_capital: invariant for ROI% calculation

    Returns:
        ComboMetrics with all fields populated.
    """
    from reward import ComboMetrics

    n = len(sim_pnl)
    if n == 0:
        return ComboMetrics()

    metrics = ComboMetrics()
    metrics.trade_date_count = n

    total_pnl = float(np.sum(sim_pnl))
    daily_roi = sim_pnl / per_unit_capital * 100.0
    total_roi = float(np.sum(daily_roi))

    metrics.total_pnl_pts = round(total_pnl, POINTS_PRECISION)
    metrics.total_roi_pct = round(total_roi, PCT_PRECISION)
    metrics.avg_roi_pct   = round(total_roi / n, PCT_PRECISION)

    winning = int(np.sum(sim_win))
    metrics.winning_trades = winning
    metrics.losing_trades  = n - winning
    metrics.win_rate_pct   = round(winning / n * 100, PCT_PRECISION)

    worst_day = float(np.min(sim_pnl))
    metrics.max_drawdown_pts = round(worst_day, POINTS_PRECISION)
    metrics.max_drawdown_pct = round(
        worst_day / per_unit_capital * 100, PCT_PRECISION
    )

    metrics.pt_exit_count   = int(np.sum(exit_type == EXIT_PT_IDX))
    metrics.sl_exit_count   = int(np.sum(exit_type == EXIT_SL_IDX))
    metrics.tsl_exit_count  = int(np.sum(exit_type == EXIT_TSL_IDX))
    metrics.univ_exit_count = int(np.sum(exit_type == EXIT_UNIV_IDX))

    return metrics


# ############################################################
#
#  PART 2: SEQUENTIAL PYTHON REPLAY
#  Used ONCE to re-replay the winning combo for detailed
#  per-day results (exit times, peak/trough times, etc.)
#  needed by db_writer.py.
#
# ############################################################


# ============================================================
# RESULT STRUCTURE
# ============================================================

def _make_result(
    sim_pnl_pts:       float,
    exit_type:         str,
    exit_time:         str,
    peak_pnl_pts:      float,
    peak_time:         str,
    trough_pnl_pts:    float,
    trough_time:       str,
    candles_processed: int,
) -> dict:
    """
    Build the result dict returned by replay_trade().
    All PNL values in points.
    INR conversion is NOT done here.
    """
    return {
        "sim_pnl_pts":       round(sim_pnl_pts, POINTS_PRECISION),
        "sim_win":           sim_pnl_pts > 0,
        "exit_type":         exit_type,
        "exit_time":         exit_time,
        "peak_pnl_pts":      round(peak_pnl_pts, POINTS_PRECISION),
        "peak_time":         peak_time,
        "trough_pnl_pts":    round(trough_pnl_pts, POINTS_PRECISION),
        "trough_time":       trough_time,
        "candles_processed": candles_processed,
    }


# ============================================================
# UNIVERSAL EXIT TIME HELPER (sequential version)
# ============================================================

def _is_universal_exit_candle(
    candle_time: str,
    universal_exit_time: str,
) -> bool:
    """
    Check if this candle's time matches or has passed
    the universal exit time.
    """
    return (
        _time_str_to_minutes(candle_time)
        >= _time_str_to_minutes(universal_exit_time)
    )


# ============================================================
# CORE SEQUENTIAL REPLAY FUNCTION
# ============================================================

def replay_trade(
    candles_pts:         list[dict],
    sl_pts:              float,
    tsl_activation_pts:  float,
    tsl_gap_pts:         float,
    pt_pts:              float,
    universal_exit_time: str,
) -> dict:
    """
    Replay one trade date with one fixed combo.
    Returns simulated result dict with full detail
    (exit time, peak time, trough time, etc.).

    Used ONCE per strategy to re-replay the winning combo
    for the detailed per-day results that db_writer.py needs.

    Args:
        candles_pts:         list of point candles
                             each: {c_pts, h_pts, l_pts, time}
        sl_pts:              stop loss level in points (positive)
        tsl_activation_pts:  points profit at which TSL arms
        tsl_gap_pts:         gap between peak and TSL floor
        pt_pts:              profit target in points
        universal_exit_time: "14:30"|"14:45"|"15:00"|"15:10"

    Returns:
        {
            sim_pnl_pts, sim_win, exit_type, exit_time,
            peak_pnl_pts, peak_time, trough_pnl_pts,
            trough_time, candles_processed
        }

    Exit priority per candle: PT > SL > TSL > UNIVERSAL
    """
    if not candles_pts:
        return _make_result(
            sim_pnl_pts=0.0,
            exit_type=EXIT_UNIVERSAL,
            exit_time="",
            peak_pnl_pts=0.0,
            peak_time="",
            trough_pnl_pts=0.0,
            trough_time="",
            candles_processed=0,
        )

    current_peak_pts:  float = 0.0
    peak_time:         str   = candles_pts[0]["time"]
    trough_pts:        float = 0.0
    trough_time:       str   = candles_pts[0]["time"]
    tsl_armed:         bool  = False
    sl_neg:            float = -sl_pts

    for i, candle in enumerate(candles_pts):
        h   = candle["h_pts"]
        l   = candle["l_pts"]
        c   = candle["c_pts"]
        t   = candle["time"]

        if h > current_peak_pts:
            current_peak_pts = h
            peak_time = t

        if l < trough_pts:
            trough_pts = l
            trough_time = t

        if not tsl_armed and current_peak_pts >= tsl_activation_pts:
            tsl_armed = True

        if tsl_armed:
            tsl_floor_pts = round(
                current_peak_pts - tsl_gap_pts,
                POINTS_PRECISION
            )
        else:
            tsl_floor_pts = None

        # Priority 1: Profit Target
        if h >= pt_pts:
            return _make_result(
                sim_pnl_pts=pt_pts,
                exit_type=EXIT_PT,
                exit_time=t,
                peak_pnl_pts=current_peak_pts,
                peak_time=peak_time,
                trough_pnl_pts=trough_pts,
                trough_time=trough_time,
                candles_processed=i + 1,
            )

        # Priority 2: Stop Loss
        if l <= sl_neg:
            return _make_result(
                sim_pnl_pts=sl_neg,
                exit_type=EXIT_SL,
                exit_time=t,
                peak_pnl_pts=current_peak_pts,
                peak_time=peak_time,
                trough_pnl_pts=trough_pts,
                trough_time=trough_time,
                candles_processed=i + 1,
            )

        # Priority 3: TSL breach
        if tsl_armed and l <= tsl_floor_pts:
            return _make_result(
                sim_pnl_pts=tsl_floor_pts,
                exit_type=EXIT_TSL,
                exit_time=t,
                peak_pnl_pts=current_peak_pts,
                peak_time=peak_time,
                trough_pnl_pts=trough_pts,
                trough_time=trough_time,
                candles_processed=i + 1,
            )

        # Priority 4: Universal exit time
        if _is_universal_exit_candle(t, universal_exit_time):
            return _make_result(
                sim_pnl_pts=c,
                exit_type=EXIT_UNIVERSAL,
                exit_time=t,
                peak_pnl_pts=current_peak_pts,
                peak_time=peak_time,
                trough_pnl_pts=trough_pts,
                trough_time=trough_time,
                candles_processed=i + 1,
            )

    # Fell through all candles — use last close as UNIVERSAL
    last = candles_pts[-1]
    return _make_result(
        sim_pnl_pts=last["c_pts"],
        exit_type=EXIT_UNIVERSAL,
        exit_time=last["time"],
        peak_pnl_pts=current_peak_pts,
        peak_time=peak_time,
        trough_pnl_pts=trough_pts,
        trough_time=trough_time,
        candles_processed=len(candles_pts),
    )


# ============================================================
# BATCH SEQUENTIAL REPLAY
# Used ONCE to re-replay the winning combo for detailed
# per-day results needed by db_writer.py.
# ============================================================

def replay_combo(
    normalised_days:     list[dict],
    sl_pts:              float,
    tsl_activation_pts:  float,
    tsl_gap_pts:         float,
    pt_pts:              float,
    universal_exit_time: str,
) -> list[dict]:
    """
    Replay one fixed combo across all normalised trade dates.
    Returns detailed per-day results (exit times, peak/trough
    times, candles processed) for db_writer.py.

    Args:
        normalised_days:  list of NormalisedDay dicts sorted ASC
        sl_pts, tsl_activation_pts, tsl_gap_pts, pt_pts,
        universal_exit_time: the fixed combo being tested

    Returns:
        List of result dicts — one per trade date.
    """
    results = []

    for day in normalised_days:
        result = replay_trade(
            candles_pts=day["candles_pts"],
            sl_pts=sl_pts,
            tsl_activation_pts=tsl_activation_pts,
            tsl_gap_pts=tsl_gap_pts,
            pt_pts=pt_pts,
            universal_exit_time=universal_exit_time,
        )
        result["trade_date"] = day["trade_date"]
        results.append(result)

    return results
