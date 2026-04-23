# main.py
# ============================================================
# Orchestrator. Entry point for the Exit Optimiser engine.
# Runs once per GitHub Actions trigger (scheduled or manual).
#
# Flow:
#   1. Initialise logging and Supabase client
#   2. Parse command line arguments for horizontal scaling (chunks)
#   3. Load all active strategies
#   4. Slice the strategies list based on the assigned chunk
#   5. Load all lot size maps (once, shared across strategies)
#   6. For each assigned active strategy:
#       a. Load strategy PNL data
#       b. Compute PER_UNIT_CAPITAL and StrategyBoundaries
#       c. Normalise all trade dates
#       d. Log search space diagnostics
#       e. Run optimiser → winner
#       f. Write winner to Supabase
#       g. Log strategy summary
#   7. Log full run summary
#
# Each strategy is completely isolated — no data, no state,
# no metrics cross strategy boundaries.
# ============================================================

import logging
import os
import sys
import traceback
import argparse
import math
from datetime import date, datetime
from typing import Optional

from supabase import Client

from config import (
    ACTIVE_STATUSES,
    StrategyBoundaries,
)
from db_loader import (
    get_client,
    load_active_strategies,
    load_all_lot_size_maps,
    load_strategy_data,
)
from normaliser import (
    compute_per_unit_capital,
    get_current_lot_size,
    normalise_all_days,
)
from optimiser import (
    run_optimiser,
    log_search_space,
)
from db_writer import write_results


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(chunk: int, total_chunks: int) -> None:
    """
    Configure logging for the full run.
    Writes to stdout (captured by GitHub Actions logs)
    and to a local file for debugging.
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | "
        "%(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"optimiser_run_chunk{chunk}_{date.today()}.log",
            mode="w",
            encoding="utf-8",
        ),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )

    # suppress verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ============================================================
# RUN DATE
# Shared across all strategies in one run.
# ============================================================

RUN_DATE: str = date.today().isoformat()   # "YYYY-MM-DD"


# ============================================================
# PER-STRATEGY PIPELINE
# ============================================================

def process_strategy(
    client:         Client,
    strategy:       dict,
    lot_size_maps:  dict[str, list[dict]],
) -> dict:
    """
    Run the full optimisation pipeline for one strategy.
    Returns a result summary dict for the run report.
    """
    strategy_id   = strategy["strategy_id"]
    strategy_name = strategy.get("strategy_name", "Unknown")

    result_base = {
        "strategy_id":   strategy_id,
        "strategy_name": strategy_name,
        "status":        "error",
        "reason":        "",
        "combo_label":   "",
        "total_roi_pct": None,
        "win_rate_pct":  None,
        "combos_tested": 0,
        "trade_dates":   0,
    }

    try:
        logger.info(
            f"{'='*60}\n"
            f"Processing strategy {strategy_id}: "
            f"{strategy_name}"
        )

        # ---- a. load PNL and lot size data -----------------
        data = load_strategy_data(
            client=client,
            strategy=strategy,
            lot_size_maps=lot_size_maps,
        )
        if not data:
            result_base["status"] = "skipped"
            result_base["reason"] = (
                "No PNL data or lot size data available."
            )
            logger.warning(
                f"[Strategy {strategy_id}] Skipped: "
                f"{result_base['reason']}"
            )
            return result_base

        lot_size_rows  = data["lot_size_rows"]
        pnl_by_date    = data["pnl_by_date"]
        la_pnl_by_date = data["la_pnl_by_date"]

        # ---- b. compute PER_UNIT_CAPITAL -------------------
        strategy_capital  = strategy["capital"]
        current_lot_size  = get_current_lot_size(lot_size_rows)
        per_unit_capital  = compute_per_unit_capital(
            strategy_capital=strategy_capital,
            current_lot_size=current_lot_size,
        )

        logger.info(
            f"[Strategy {strategy_id}] "
            f"STRATEGY_CAPITAL={strategy_capital} | "
            f"current_lot_size={current_lot_size} | "
            f"PER_UNIT_CAPITAL={per_unit_capital}"
        )

        # ---- c. build StrategyBoundaries -------------------
        is_buying = (
            strategy.get("trades_type") == "Option Buying"
            or strategy.get("side") == "Buy"
        )

        boundaries = StrategyBoundaries(
            per_unit_capital=per_unit_capital,
            is_buying=is_buying,
        )

        # ---- d. normalise all trade dates ------------------
        normalised_days = normalise_all_days(
            pnl_by_date=pnl_by_date,
            lot_size_rows=lot_size_rows,
            per_unit_capital=per_unit_capital,
            la_pnl_by_date=la_pnl_by_date,
        )

        if not normalised_days:
            result_base["status"] = "skipped"
            result_base["reason"] = (
                "All trade dates failed normalisation."
            )
            logger.warning(
                f"[Strategy {strategy_id}] Skipped: "
                f"{result_base['reason']}"
            )
            return result_base

        n_days = len(normalised_days)
        result_base["trade_dates"] = n_days

        # ---- e. log search space before optimising ---------
        log_search_space(
            strategy_id=strategy_id,
            boundaries=boundaries,
            current_lot=current_lot_size,
            n_days=n_days,
        )

        # ---- f. run optimiser ------------------------------
        winner = run_optimiser(
            strategy_id=strategy_id,
            normalised_days=normalised_days,
            per_unit_capital=per_unit_capital,
            boundaries=boundaries,
        )

        if not winner:
            result_base["status"] = "error"
            result_base["reason"] = (
                "Optimiser returned no valid winner. "
                "Check boundary config — may be too tight."
            )
            logger.error(
                f"[Strategy {strategy_id}] "
                f"{result_base['reason']}"
            )
            return result_base

        result_base["combos_tested"] = winner["combos_tested"]

        # ---- g. write to Supabase --------------------------
        write_ok = write_results(
            client=client,
            strategy_id=strategy_id,
            winner=winner,
            normalised_days=normalised_days,
            current_lot_size=current_lot_size,
            per_unit_capital=per_unit_capital,
            run_date=RUN_DATE,
        )

        if not write_ok:
            result_base["status"] = "error"
            result_base["reason"] = (
                "Database write failed. "
                "Check Supabase logs."
            )
            return result_base

        # ---- h. populate success result --------------------
        train_metrics = winner["train_metrics"]
        combo_label   = (
            winner.get("combo_label")
            or f"SL {winner['sl_pts']}pts | "
               f"TSL {winner['tsl_activation_pts']}pts | "
               f"PT {winner['pt_pts']}pts"
        )

        result_base.update({
            "status":        "success",
            "reason":        "",
            "combo_label":   combo_label,
            "total_roi_pct": train_metrics.total_roi_pct,
            "win_rate_pct":  train_metrics.win_rate_pct,
        })

        logger.info(
            f"[Strategy {strategy_id}] "
            f"Completed successfully.\n"
            f"  Combo:    {combo_label}\n"
            f"  ROI:      {train_metrics.total_roi_pct}%\n"
            f"  Win rate: {train_metrics.win_rate_pct}%\n"
            f"  Drawdown: {train_metrics.max_drawdown_pct}%\n"
            f"  Tested:   {winner['combos_tested']} combos | "
            f"{winner['combos_valid']} valid\n"
            f"  Dates:    {winner['train_dates']} train | "
            f"{winner['validation_dates']} validation"
        )

        return result_base

    except Exception as e:
        logger.error(
            f"[Strategy {strategy_id}] "
            f"Unexpected error: {e}\n"
            f"{traceback.format_exc()}"
        )
        result_base["status"] = "error"
        result_base["reason"] = str(e)
        return result_base


# ============================================================
# RUN SUMMARY LOGGER
# ============================================================

def log_run_summary(
    results:    list[dict],
    start_time: datetime,
) -> None:
    """
    Log a structured summary of the full run after all
    strategies have been processed.
    """
    elapsed = round(
        (datetime.now() - start_time).total_seconds(), 1
    )

    success  = [r for r in results if r["status"] == "success"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    errors   = [r for r in results if r["status"] == "error"]

    logger.info(
        f"\n{'='*60}\n"
        f"RUN SUMMARY — {RUN_DATE}\n"
        f"{'='*60}\n"
        f"Total strategies processed in this chunk: {len(results)}\n"
        f"  Succeeded:      {len(success)}\n"
        f"  Skipped:        {len(skipped)}\n"
        f"  Errors:         {len(errors)}\n"
        f"Elapsed:          {elapsed}s\n"
        f"{'='*60}"
    )

    if success:
        logger.info("SUCCESSFUL STRATEGIES:")
        for r in success:
            logger.info(
                f"  [{r['strategy_id']}] "
                f"{r['strategy_name']}\n"
                f"    Combo:    {r['combo_label']}\n"
                f"    ROI:      {r['total_roi_pct']}% | "
                f"Win rate: {r['win_rate_pct']}% | "
                f"Tested: {r['combos_tested']} combos | "
                f"Dates: {r['trade_dates']}"
            )

    if skipped:
        logger.warning("SKIPPED STRATEGIES:")
        for r in skipped:
            logger.warning(
                f"  [{r['strategy_id']}] "
                f"{r['strategy_name']}: {r['reason']}"
            )

    if errors:
        logger.error("FAILED STRATEGIES:")
        for r in errors:
            logger.error(
                f"  [{r['strategy_id']}] "
                f"{r['strategy_name']}: {r['reason']}"
            )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """
    Main entry point. Called by GitHub Actions.
    """
    parser = argparse.ArgumentParser(description="Run Exit Optimiser")
    parser.add_argument("--chunk", type=int, default=1, help="Which chunk to run (1-indexed)")
    parser.add_argument("--total-chunks", type=int, default=1, help="Total number of chunks")
    args = parser.parse_args()

    setup_logging(args.chunk, args.total_chunks)
    start_time = datetime.now()

    logger.info(
        f"\n{'='*60}\n"
        f"EXIT OPTIMISER — RUN STARTED (Chunk {args.chunk}/{args.total_chunks})\n"
        f"Date: {RUN_DATE}\n"
        f"{'='*60}"
    )

    # ---- initialise Supabase client ------------------------
    try:
        client = get_client()
        logger.info("Supabase client initialised.")
    except EnvironmentError as e:
        logger.critical(
            f"Cannot connect to Supabase: {e}\n"
            f"Ensure SUPABASE_URL and SUPABASE_SERVICE_KEY "
            f"are set as GitHub Actions secrets."
        )
        sys.exit(1)

    # ---- load shared data (once for all strategies) --------
    try:
        strategies = load_active_strategies(client)
    except Exception as e:
        logger.critical(
            f"Failed to load strategies: {e}\n"
            f"{traceback.format_exc()}"
        )
        sys.exit(1)

    if not strategies:
        logger.warning(
            "No active strategies found. Nothing to optimise."
        )
        sys.exit(0)

    # ---- HORIZONTAL SCALING: Slice the strategies array ----
    total_strategies = len(strategies)
    chunk_size = math.ceil(total_strategies / args.total_chunks)
    start_idx = (args.chunk - 1) * chunk_size
    end_idx = start_idx + chunk_size
    
    strategies = strategies[start_idx:end_idx]

    if not strategies:
        logger.info(f"Chunk {args.chunk} has no strategies to process. Exiting cleanly.")
        sys.exit(0)

    logger.info(
        f"Chunk {args.chunk}/{args.total_chunks}: "
        f"Processing {len(strategies)} of {total_strategies} total active strategies."
    )

    try:
        lot_size_maps = load_all_lot_size_maps(client)
    except Exception as e:
        logger.critical(
            f"Failed to load lot size maps: {e}\n"
            f"{traceback.format_exc()}"
        )
        sys.exit(1)

    # ---- process each strategy -----------------------------
    results: list[dict] = []

    for i, strategy in enumerate(strategies, start=1):
        logger.info(
            f"\nStrategy {i} of {len(strategies)} in this chunk: "
            f"ID={strategy['strategy_id']} | "
            f"{strategy.get('strategy_name', 'Unknown')}"
        )

        result = process_strategy(
            client=client,
            strategy=strategy,
            lot_size_maps=lot_size_maps,
        )
        results.append(result)

    # ---- log full run summary ------------------------------
    log_run_summary(results, start_time)

    # ---- exit code -----------------------------------------
    error_count = sum(
        1 for r in results if r["status"] == "error"
    )

    if error_count > 0:
        logger.error(
            f"{error_count} strategy/strategies failed. "
            f"Check logs above for details."
        )
        sys.exit(1)

    logger.info("All assigned strategies completed. Exiting cleanly.")
    sys.exit(0)


# ============================================================
# SCRIPT ENTRY
# ============================================================

if __name__ == "__main__":
    main()
