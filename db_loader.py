# db_loader.py
# ============================================================
# Loads all data from Supabase needed by the optimiser.
# Three responsibilities:
#   1. Load active strategies (selected columns only)
#   2. Build the lot size monthly map per instrument
#   3. Load pnl_data JSONB per strategy and parse candles
#
# This file only READS from Supabase.
# Nothing is written here — that is db_writer.py.
#
# Capital vocabulary used in this file:
#   STRATEGY_CAPITAL  — loaded from strategies.capital
#   PER_UNIT_CAPITAL  — computed here once per strategy
#   HISTORICAL_CAPITAL — NOT computed here, that is
#                        normaliser.py's responsibility
# ============================================================

import os
import json
import logging
from datetime import date, datetime
from typing import Any

from supabase import create_client, Client
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    STRATEGY_COLUMNS,
    ACTIVE_STATUSES,
    SUPPORTED_INSTRUMENTS,
)

logger = logging.getLogger(__name__)


# ============================================================
# TYPE ALIASES
# ============================================================

# One parsed candle: {c, h, l, time} all as floats/strings
Candle = dict[str, Any]

# All candles for one trade date
DayCandles = dict[str, list[Candle]]   # key: trade_date str

# Full PNL dataset for one strategy
StrategyPNL = dict[str, DayCandles]    # key: strategy_id str


# ============================================================
# CLIENT
# ============================================================

def get_client() -> Client:
    """
    Create and return Supabase client.
    Reads credentials from environment variables —
    never from config.py directly in production.
    """
    url = os.environ.get("SUPABASE_URL") or SUPABASE_URL
    key = os.environ.get("SUPABASE_SERVICE_KEY") or SUPABASE_KEY

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set "
            "as environment variables."
        )
    return create_client(url, key)


# ============================================================
# 1. LOAD ACTIVE STRATEGIES
# ============================================================

def load_active_strategies(client: Client) -> list[dict]:
    """
    Load active strategies from the strategies table.
    Returns only the columns defined in config.STRATEGY_COLUMNS.
    Filters to config.ACTIVE_STATUSES only.

    Returns:
        List of strategy dicts. Each dict contains:
            strategy_id       (int)
            strategy_name     (str)
            deployment_type   (str)
            la_mapping_id     (str | None)
            capital           (float)  ← this is STRATEGY_CAPITAL
            index_name        (str)
            trades_type       (str)
            option_expiry     (str)
            side              (str)
            status            (str)
    """
    logger.info("Loading active strategies from Supabase...")

    response = (
        client.table("strategies")
        .select(", ".join(STRATEGY_COLUMNS))
        .in_("status", ACTIVE_STATUSES)
        .in_("index_name", SUPPORTED_INSTRUMENTS)
        .execute()
    )

    strategies = response.data or []

    # coerce types — capital must be float, strategy_id must be int
    for s in strategies:
        s["strategy_id"] = int(s["strategy_id"])
        s["capital"] = float(s["capital"]) if s.get("capital") else None
        s["la_mapping_id"] = s.get("la_mapping_id") or None

    # drop any strategy with no capital — cannot compute PER_UNIT_CAPITAL
    valid = [s for s in strategies if s["capital"]]
    dropped = len(strategies) - len(valid)
    if dropped:
        logger.warning(
            f"Dropped {dropped} strategies with null capital."
        )

    logger.info(f"Loaded {len(valid)} active strategies.")
    return valid


# ============================================================
# 2. LOT SIZE MONTHLY MAP
# ============================================================

def load_lot_size_map(
    client: Client,
    instrument: str
) -> list[dict]:
    """
    Load all lot_size rows for one instrument, ordered by
    effective_date descending.

    The monthly map lookup works as follows:
        For any given trade_date, find the row where
        effective_date is the most recent date that is
        <= the first day of trade_date's month.

    This function loads the raw rows. The lookup itself
    is performed by normaliser.get_lot_size_for_date().

    Returns:
        List of dicts ordered by effective_date DESC:
            [{"lot_size": 65, "effective_date": "2025-12-31"}, ...]
    """
    logger.info(f"Loading lot size map for {instrument}...")

    response = (
        client.table("lot_sizes")
        .select("lot_size, effective_date")
        .eq("instrument", instrument)
        .order("effective_date", desc=True)
        .execute()
    )

    rows = response.data or []

    # coerce types
    for row in rows:
        row["lot_size"] = int(row["lot_size"])
        row["effective_date"] = datetime.strptime(
            row["effective_date"], "%Y-%m-%d"
        ).date()

    logger.info(
        f"Loaded {len(rows)} lot size rows for {instrument}."
    )
    return rows


def load_all_lot_size_maps(client: Client) -> dict[str, list[dict]]:
    """
    Load lot size maps for all supported instruments.

    Returns:
        {
            "NIFTY":     [...rows ordered by effective_date DESC],
            "BANKNIFTY": [...rows ordered by effective_date DESC],
        }
    """
    return {
        instrument: load_lot_size_map(client, instrument)
        for instrument in SUPPORTED_INSTRUMENTS
    }


# ============================================================
# 3. LOAD PNL DATA
# ============================================================

def _parse_pnl_data(raw_jsonb: Any) -> list[Candle]:
    """
    Parse the pnl_data JSONB column for one trade date row.

    Input from Supabase is a list of dicts with string values:
        [{"c": "168.75", "h": "311.25", "l": "-135.0",
          "time": "9:33 AM"}, ...]

    Output is a list of dicts with numeric c/h/l and str time:
        [{"c": 168.75, "h": 311.25, "l": -135.0,
          "time": "9:33 AM"}, ...]

    Candles where c == h == l == 0.0 are pre-open noise
    and are filtered out (e.g. 9:31 AM, 9:32 AM stubs).
    """
    if isinstance(raw_jsonb, str):
        candles = json.loads(raw_jsonb)
    else:
        candles = raw_jsonb or []

    parsed = []
    for candle in candles:
        c = float(candle.get("c", 0))
        h = float(candle.get("h", 0))
        l = float(candle.get("l", 0))
        t = candle.get("time", "")

        # skip pre-open zero candles
        if c == 0.0 and h == 0.0 and l == 0.0:
            continue

        parsed.append({"c": c, "h": h, "l": l, "time": t})

    return parsed


def load_pnl_for_strategy(
    client: Client,
    strategy_id: int,
) -> dict[str, list[Candle]]:
    """
    Load all pnl rows for one strategy from
    intraday_pnl_1min_ohlc and parse each day's candles.

    Returns:
        {
            "2025-01-13": [
                {"c": 168.75, "h": 311.25,
                 "l": -135.0, "time": "9:33 AM"},
                ...
            ],
            "2025-01-14": [...],
            ...
        }

    Trade dates are sorted ascending (oldest first)
    so walk-forward split works correctly.
    """
    logger.info(
        f"Loading PNL data for strategy {strategy_id}..."
    )

    # paginate to handle strategies with many trade dates
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            client.table("intraday_pnl_1min_ohlc")
            .select("trade_date, pnl_data")
            .eq("strategy_id", strategy_id)
            .order("trade_date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        all_rows.extend(batch)

        if len(batch) < page_size:
            break
        offset += page_size

    if not all_rows:
        logger.warning(
            f"No PNL data found for strategy {strategy_id}."
        )
        return {}

    # parse each row
    result: dict[str, list[Candle]] = {}
    for row in all_rows:
        trade_date_str = str(row["trade_date"])
        candles = _parse_pnl_data(row["pnl_data"])
        if candles:
            result[trade_date_str] = candles

    logger.info(
        f"Strategy {strategy_id}: loaded "
        f"{len(result)} trade dates."
    )
    return result


def load_pnl_for_la_strategy(
    client: Client,
    la_mapping_id: str,
) -> dict[str, list[Candle]]:
    """
    Load PNL data for the Live Auto strategy linked via
    la_mapping_id.

    la_mapping_id in the strategies table contains the
    strategy_id of the Live Auto deployment.
    Returns same structure as load_pnl_for_strategy().
    Returns empty dict if la_mapping_id is None.
    """
    if not la_mapping_id:
        return {}

    try:
        la_id = int(la_mapping_id)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid la_mapping_id: {la_mapping_id}. "
            f"Skipping LA PNL load."
        )
        return {}

    return load_pnl_for_strategy(client, la_id)


# ============================================================
# 4. CONVENIENCE LOADER
# Loads everything needed for one strategy in one call.
# Called by main.py per strategy.
# ============================================================

def load_strategy_data(
    client: Client,
    strategy: dict,
    lot_size_maps: dict[str, list[dict]],
) -> dict:
    """
    Load all data required to run the optimiser for one
    strategy. Returns a single dict passed to the engine.

    Args:
        client:          Supabase client
        strategy:        One row from load_active_strategies()
        lot_size_maps:   Output of load_all_lot_size_maps()

    Returns:
        {
            "strategy":      strategy dict,
            "lot_size_rows": [...] for this strategy's instrument,
            "pnl_by_date":   {"2025-01-13": [candles...], ...},
            "la_pnl_by_date":{"2025-01-13": [candles...], ...}
                             or {} if no la_mapping_id,
        }
    """
    strategy_id  = strategy["strategy_id"]
    instrument   = strategy["index_name"]
    la_mapping   = strategy.get("la_mapping_id")

    lot_size_rows = lot_size_maps.get(instrument, [])
    if not lot_size_rows:
        logger.warning(
            f"No lot size rows found for instrument "
            f"{instrument}. Strategy {strategy_id} "
            f"will be skipped."
        )
        return {}

    pnl_by_date = load_pnl_for_strategy(client, strategy_id)
    if not pnl_by_date:
        logger.warning(
            f"Strategy {strategy_id} has no PNL data. "
            f"Skipping."
        )
        return {}

    la_pnl_by_date = load_pnl_for_la_strategy(
        client, la_mapping
    )

    return {
        "strategy":       strategy,
        "lot_size_rows":  lot_size_rows,
        "pnl_by_date":    pnl_by_date,
        "la_pnl_by_date": la_pnl_by_date,
    }
