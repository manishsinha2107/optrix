# normaliser.py
# ============================================================
# Responsible for all lot-size aware calculations.
# Three responsibilities:
#   1. Compute PER_UNIT_CAPITAL from STRATEGY_CAPITAL
#   2. Resolve the correct lot size for any trade date
#   3. Convert candle PNL from INR to points for simulation
#      and compute HISTORICAL_CAPITAL per trade date for output
#
# Capital vocabulary enforced in this file:
#   STRATEGY_CAPITAL   — input only, used once, never stored
#   PER_UNIT_CAPITAL   — the invariant, computed and returned
#   HISTORICAL_CAPITAL — computed per trade date for output only
#
# The simulation engine ONLY works in points.
# INR values are only produced at the very end for storage.
# ============================================================

import logging
from datetime import date, datetime
from typing import Optional

from config import (
    POINTS_PRECISION,
    INR_PRECISION,
    PCT_PRECISION,
    StrategyBoundaries,
)

logger = logging.getLogger(__name__)


# ============================================================
# TYPE ALIASES
# ============================================================

# One candle in points (converted from INR)
CandlePts = dict  # {c_pts, h_pts, l_pts, time}

# One trade date fully normalised — ready for simulator
NormalisedDay = dict  # {trade_date, lot_size, historical_capital,
                      #  candles_pts, actual_pnl_inr,
                      #  actual_roi_pct, per_unit_capital}


# ============================================================
# 1. PER_UNIT_CAPITAL
# ============================================================

def compute_per_unit_capital(
    strategy_capital: float,
    current_lot_size: int,
) -> float:
    """
    Compute PER_UNIT_CAPITAL from STRATEGY_CAPITAL.
    This is the invariant that anchors all boundary
    calculations and ROI output.

    Formula:
        PER_UNIT_CAPITAL = STRATEGY_CAPITAL / current_lot_size

    Example:
        STRATEGY_CAPITAL = 65000, current_lot_size = 65
        PER_UNIT_CAPITAL = 1000.0

    Args:
        strategy_capital: raw capital from strategies.capital
                          (this is STRATEGY_CAPITAL — used
                           here and never again)
        current_lot_size: lot size active today for this
                          instrument (latest effective_date row)

    Returns:
        PER_UNIT_CAPITAL as float
    """
    if current_lot_size <= 0:
        raise ValueError(
            f"current_lot_size must be > 0, "
            f"got {current_lot_size}"
        )
    if strategy_capital <= 0:
        raise ValueError(
            f"STRATEGY_CAPITAL must be > 0, "
            f"got {strategy_capital}"
        )

    per_unit_capital = strategy_capital / current_lot_size

    logger.debug(
        f"STRATEGY_CAPITAL={strategy_capital} / "
        f"current_lot_size={current_lot_size} → "
        f"PER_UNIT_CAPITAL={per_unit_capital}"
    )
    return per_unit_capital


# ============================================================
# 2. LOT SIZE LOOKUP
# ============================================================

def get_lot_size_for_date(
    trade_date: str,
    lot_size_rows: list[dict],
) -> int:
    """
    Resolve the correct lot size for a given trade date.

    Logic:
        Find the row in lot_size_rows where effective_date
        is the most recent date <= the 1st of trade_date's
        month. lot_size_rows must be pre-sorted by
        effective_date DESC (db_loader guarantees this).

    Example:
        trade_date = "2026-01-15"
        → first_of_month = 2026-01-01
        → scan rows DESC for effective_date <= 2026-01-01
        → finds 2025-12-31 → lot_size = 65

        trade_date = "2025-11-20"
        → first_of_month = 2025-11-01
        → scan rows DESC for effective_date <= 2025-11-01
        → finds 2024-11-21 → lot_size = 75

    Args:
        trade_date:    "YYYY-MM-DD" string
        lot_size_rows: list of dicts sorted by effective_date
                       DESC, each with keys:
                       {"lot_size": int, "effective_date": date}

    Returns:
        lot_size as int

    Raises:
        ValueError if no matching lot size found
    """
    # derive first of month from trade_date
    td = datetime.strptime(trade_date, "%Y-%m-%d").date()
    first_of_month = td.replace(day=1)

    # rows are DESC — first match is the most recent valid row
    for row in lot_size_rows:
        if row["effective_date"] <= first_of_month:
            return row["lot_size"]

    raise ValueError(
        f"No lot size found for trade_date={trade_date} "
        f"(first_of_month={first_of_month}). "
        f"Check lot_sizes table for {td.year}-{td.month:02d}."
    )


def get_current_lot_size(lot_size_rows: list[dict]) -> int:
    """
    Get the current (most recent) lot size.
    Since lot_size_rows is sorted DESC, the first row
    is always the current lot size.

    Used once at startup to compute PER_UNIT_CAPITAL.
    """
    if not lot_size_rows:
        raise ValueError("lot_size_rows is empty.")
    return lot_size_rows[0]["lot_size"]


# ============================================================
# 3. HISTORICAL_CAPITAL PER TRADE DATE
# ============================================================

def compute_historical_capital(
    per_unit_capital: float,
    lot_size_on_date: int,
) -> float:
    """
    Compute HISTORICAL_CAPITAL for one specific trade date.

    Formula:
        HISTORICAL_CAPITAL = PER_UNIT_CAPITAL × lot_size_on_date

    Example:
        PER_UNIT_CAPITAL = 1000.0, lot_size_on_date = 75
        HISTORICAL_CAPITAL = 75000.0  (Dec 2025 trade)

        PER_UNIT_CAPITAL = 1000.0, lot_size_on_date = 65
        HISTORICAL_CAPITAL = 65000.0  (Jan 2026 trade)

    IMPORTANT: This value is computed per trade date ONLY
    for the sim_daily output table.
    It is NEVER passed into the simulation loop.
    The simulation loop works entirely in points.

    Args:
        per_unit_capital: the invariant computed at startup
        lot_size_on_date: lot size for this specific trade date

    Returns:
        HISTORICAL_CAPITAL as float
    """
    return round(per_unit_capital * lot_size_on_date, INR_PRECISION)


# ============================================================
# 4. INR → POINTS CONVERSION
# ============================================================

def inr_to_pts(
    inr_value: float,
    lot_size_on_date: int,
) -> float:
    """
    Convert one INR PNL value to points using the lot size
    active on that trade date.

    Formula:
        points = inr_value / lot_size_on_date

    Example:
        inr_value = 1625.0, lot_size_on_date = 65
        → points = 25.0

        inr_value = 1875.0, lot_size_on_date = 75
        → points = 25.0

    The same 25 points regardless of which month traded.
    This is the core lot-size agnostic conversion.
    """
    if lot_size_on_date <= 0:
        raise ValueError(
            f"lot_size_on_date must be > 0, "
            f"got {lot_size_on_date}"
        )
    return round(inr_value / lot_size_on_date, POINTS_PRECISION)


def pts_to_inr(
    pts_value: float,
    lot_size: int,
) -> float:
    """
    Convert points back to INR using a given lot size.

    Used ONLY at output stage:
        - sim_pnl_inr = sim_pnl_pts × lot_size_on_date
        - combo INR labels = winning_pts × current_lot_size

    Never called inside the simulation loop.
    """
    return round(pts_value * lot_size, INR_PRECISION)


# ============================================================
# 5. CANDLE NORMALISER
# Convert one day's candles from INR to points.
# ============================================================

def normalise_candles(
    candles_inr: list[dict],
    lot_size_on_date: int,
) -> list[CandlePts]:
    """
    Convert a list of INR candles to point candles.

    Input candle:  {"c": 1625.0, "h": 1950.0,
                    "l": -325.0, "time": "10:15 AM"}
    Output candle: {"c_pts": 25.0, "h_pts": 30.0,
                    "l_pts": -5.0, "time": "10:15 AM"}

    The original INR values are not retained in the output —
    the simulator works only with _pts fields.

    Args:
        candles_inr:     parsed candles from db_loader
                         (c, h, l in INR, time as str)
        lot_size_on_date: lot size for this trade date

    Returns:
        list of point candles with c_pts, h_pts, l_pts, time
    """
    return [
        {
            "c_pts": inr_to_pts(c["c"], lot_size_on_date),
            "h_pts": inr_to_pts(c["h"], lot_size_on_date),
            "l_pts": inr_to_pts(c["l"], lot_size_on_date),
            "time":  c["time"],
        }
        for c in candles_inr
    ]


# ============================================================
# 6. ACTUAL PNL EXTRACTOR
# The actual PNL for a trade date is the last candle's
# c value (cumulative close PNL at end of day).
# Now updated to also extract peak, trough, and exit time.
# ============================================================

def extract_actual_pnl(
    candles_inr: list[dict],
) -> dict:
    """
    Extract comprehensive actual performance data from candles.

    The pnl_data JSONB stores cumulative PNL — so the last
    candle's c value is the total PNL for that trade date.
    We scan all candles to find the highest high and lowest low.

    Returns:
        dict containing: pnl_inr, peak_inr, peak_time, 
        trough_inr, trough_time, exit_time
    """
    if not candles_inr:
        return {
            "pnl_inr": None,
            "peak_inr": None,
            "peak_time": None,
            "trough_inr": None,
            "trough_time": None,
            "exit_time": None
        }

    last_candle = candles_inr[-1]
    actual_pnl_inr = round(last_candle["c"], INR_PRECISION)
    exit_time = last_candle["time"]

    peak_inr = -float('inf')
    peak_time = ""
    trough_inr = float('inf')
    trough_time = ""

    for c in candles_inr:
        if c["h"] > peak_inr:
            peak_inr = c["h"]
            peak_time = c["time"]
        if c["l"] < trough_inr:
            trough_inr = c["l"]
            trough_time = c["time"]

    return {
        "pnl_inr": actual_pnl_inr,
        "peak_inr": round(peak_inr, INR_PRECISION) if peak_inr != -float('inf') else actual_pnl_inr,
        "peak_time": peak_time,
        "trough_inr": round(trough_inr, INR_PRECISION) if trough_inr != float('inf') else actual_pnl_inr,
        "trough_time": trough_time,
        "exit_time": exit_time
    }


def compute_actual_roi_pct(
    actual_pnl_inr: Optional[float],
    historical_capital: float,
) -> Optional[float]:
    """
    Compute actual ROI% for one trade date.

    Formula:
        actual_roi_pct = actual_pnl_inr / HISTORICAL_CAPITAL × 100

    Uses HISTORICAL_CAPITAL (not PER_UNIT_CAPITAL) because
    actual PNL is in INR and must be compared to the capital
    that was actually deployed on that date.
    """
    if actual_pnl_inr is None:
        return None
    if historical_capital <= 0:
        return None
    return round(
        actual_pnl_inr / historical_capital * 100,
        PCT_PRECISION
    )


# ============================================================
# 7. LA PNL EXTRACTOR
# Same logic as actual PNL but for the Live Auto strategy.
# ============================================================

def extract_la_pnl(
    la_candles_inr: list[dict],
    trade_date: str,
    la_pnl_by_date: dict[str, list[dict]],
    historical_capital: float,
) -> dict:
    """
    Extract comprehensive LA performance data.
    Retains original signature to prevent breaking dependencies.

    Returns:
        dict containing: pnl_inr, roi_pct, peak_inr, peak_time, 
        trough_inr, trough_time, exit_time
    """
    la_candles = la_pnl_by_date.get(trade_date)
    if not la_candles:
        return {
            "pnl_inr": None,
            "roi_pct": None,
            "peak_inr": None,
            "peak_time": None,
            "trough_inr": None,
            "trough_time": None,
            "exit_time": None
        }

    la_data = extract_actual_pnl(la_candles)
    la_data["roi_pct"] = compute_actual_roi_pct(
        la_data["pnl_inr"], historical_capital
    )
    
    return la_data


# ============================================================
# 8. FULL DAY NORMALISER
# Combines all of the above into one call per trade date.
# Called by main.py during data preparation before the
# optimiser loop starts.
# ============================================================

def normalise_day(
    trade_date: str,
    candles_inr: list[dict],
    lot_size_rows: list[dict],
    per_unit_capital: float,
    la_pnl_by_date: dict[str, list[dict]],
) -> NormalisedDay:
    """
    Produce a fully normalised day ready for the simulator.

    Steps:
        1. Resolve lot_size_on_date from lot_size_rows
        2. Compute HISTORICAL_CAPITAL for this date
        3. Convert all candles from INR to points
        4. Extract actual EOD PNL, ROI%, peak, trough, and exit time
        5. Extract LA PNL, ROI%, peak, trough, and exit time if available

    Returns a dict with everything the simulator needs
    for this trade date — no further lookups required.

    Return structure:
        {
            "trade_date":        "2025-01-13",
            "lot_size_on_date":  75,
            "historical_capital": 75000.0,
            "per_unit_capital":  1000.0,
            "candles_pts":       [{c_pts, h_pts, l_pts, time}, ...],
            "actual_pnl_inr":    142.5,
            "actual_roi_pct":    0.19,
            "actual_peak_inr":   200.0,
            "actual_peak_time":  "10:15 AM",
            "actual_trough_inr": -50.0,
            "actual_trough_time":"09:30 AM",
            "actual_exit_time":  "15:15 PM",
            "la_pnl_inr":        None,
            "la_roi_pct":        None,
            "la_peak_inr":       None,
            "la_peak_time":      None,
            "la_trough_inr":     None,
            "la_trough_time":    None,
            "la_exit_time":      None,
        }
    """
    lot_size_on_date = get_lot_size_for_date(
        trade_date, lot_size_rows
    )
    historical_capital = compute_historical_capital(
        per_unit_capital, lot_size_on_date
    )
    candles_pts = normalise_candles(candles_inr, lot_size_on_date)
    
    actual_data = extract_actual_pnl(candles_inr)
    actual_roi_pct = compute_actual_roi_pct(
        actual_data["pnl_inr"], historical_capital
    )
    
    la_data = extract_la_pnl(
        candles_inr, trade_date,
        la_pnl_by_date, historical_capital
    )

    return {
        "trade_date":         trade_date,
        "lot_size_on_date":   lot_size_on_date,
        "historical_capital": historical_capital,
        "per_unit_capital":   per_unit_capital,
        "candles_pts":        candles_pts,
        
        # Expanded Actual Data
        "actual_pnl_inr":     actual_data["pnl_inr"],
        "actual_roi_pct":     actual_roi_pct,
        "actual_peak_inr":    actual_data["peak_inr"],
        "actual_peak_time":   actual_data["peak_time"],
        "actual_trough_inr":  actual_data["trough_inr"],
        "actual_trough_time": actual_data["trough_time"],
        "actual_exit_time":   actual_data["exit_time"],
        
        # Expanded LA Data
        "la_pnl_inr":         la_data["pnl_inr"],
        "la_roi_pct":         la_data["roi_pct"],
        "la_peak_inr":        la_data["peak_inr"],
        "la_peak_time":       la_data["peak_time"],
        "la_trough_inr":      la_data["trough_inr"],
        "la_trough_time":     la_data["trough_time"],
        "la_exit_time":       la_data["exit_time"],
    }


def normalise_all_days(
    pnl_by_date: dict[str, list[dict]],
    lot_size_rows: list[dict],
    per_unit_capital: float,
    la_pnl_by_date: dict[str, list[dict]],
) -> list[NormalisedDay]:
    """
    Normalise every trade date for one strategy.
    Returns a list sorted by trade_date ascending —
    oldest first, so walk-forward split works correctly.

    This is called once per strategy before the Optuna
    loop begins. The result is passed directly to the
    simulator — no further normalisation needed inside
    the optimiser loop.

    Args:
        pnl_by_date:      output of db_loader.load_pnl_for_strategy
        lot_size_rows:    output of db_loader.load_lot_size_map
        per_unit_capital: computed by compute_per_unit_capital
        la_pnl_by_date:   output of db_loader.load_pnl_for_la_strategy

    Returns:
        List of NormalisedDay dicts, sorted by trade_date ASC
    """
    days = []
    skipped = 0

    for trade_date, candles_inr in sorted(pnl_by_date.items()):
        try:
            day = normalise_day(
                trade_date=trade_date,
                candles_inr=candles_inr,
                lot_size_rows=lot_size_rows,
                per_unit_capital=per_unit_capital,
                la_pnl_by_date=la_pnl_by_date,
            )
            days.append(day)
        except ValueError as e:
            logger.warning(
                f"Skipping trade_date={trade_date}: {e}"
            )
            skipped += 1

    if skipped:
        logger.warning(
            f"Skipped {skipped} trade dates during normalisation."
        )

    logger.info(
        f"Normalised {len(days)} trade dates. "
        f"PER_UNIT_CAPITAL={per_unit_capital}"
    )
    return days
