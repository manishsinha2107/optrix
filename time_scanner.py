# time_scanner.py
import logging
import numpy as np

from config import UNIVERSAL_EXIT_TIMES, POINTS_PRECISION, PCT_PRECISION
from simulator import (
    preprocess_days,
    replay_combo_np,
    aggregate_results_np,
    replay_combo,
    _time_str_to_minutes,
)

logger = logging.getLogger(__name__)

def run_time_scanner(normalised_days: list[dict], per_unit_capital: float) -> dict:
    """
    Brute-forces the absolute highest ROI from pure time exits.
    Bypasses Optuna and risk-adjusted reward scoring entirely.
    """
    n_days = len(normalised_days)
    if n_days == 0:
        return {}

    data = preprocess_days(normalised_days)
    inf_pts = 99999.0
    
    best_time = None
    best_roi = -float('inf')
    best_metrics = None
    
    for univ_time in UNIVERSAL_EXIT_TIMES:
        univ_minutes = _time_str_to_minutes(univ_time)
        
        sim_pnl, sim_win, exit_type = replay_combo_np(
            data=data,
            sl_pts=inf_pts,
            tsl_activation_pts=inf_pts,
            tsl_gap_pts=inf_pts,
            pt_pts=inf_pts,
            universal_exit_minutes=univ_minutes,
        )
        
        metrics = aggregate_results_np(sim_pnl, sim_win, exit_type, per_unit_capital)
        
        if metrics.total_roi_pct > best_roi:
            best_roi = metrics.total_roi_pct
            best_time = univ_time
            best_metrics = metrics
            
    if not best_time:
        return {}
        
    # Replay the absolute best time sequentially to generate the daily tape for db_writer
    daily_results = replay_combo(
        normalised_days=normalised_days,
        sl_pts=inf_pts,
        tsl_activation_pts=inf_pts,
        tsl_gap_pts=inf_pts,
        pt_pts=inf_pts,
        universal_exit_time=best_time,
    )
    
    # Key by trade_date for instant lookup in db_writer
    daily_lookup = {r["trade_date"]: r for r in daily_results}
    
    logger.info(
        f"Time Scanner found highest pure mathematical ROI at {best_time} "
        f"({best_metrics.total_roi_pct}%)"
    )
    
    return {
        "time_exit_time": best_time,
        "time_exit_pnl_pts": best_metrics.total_pnl_pts,
        "time_exit_roi_pct": best_metrics.total_roi_pct,
        "time_exit_win_rate_pct": best_metrics.win_rate_pct,
        "time_exit_max_dd_pts": best_metrics.max_drawdown_pts,
        "daily_lookup": daily_lookup,
    }
