# config.py
# ============================================================
# Central configuration for the Exit Optimiser engine.
# All boundary values are expressed as percentages of
# PER_UNIT_CAPITAL and converted to points at runtime.
# Nothing in this file uses STRATEGY_CAPITAL or
# HISTORICAL_CAPITAL directly.
# ============================================================

from dataclasses import dataclass, field
from typing import List


# ============================================================
# SUPABASE CONNECTION
# Set via environment variables in GitHub Actions secrets.
# Never hardcode credentials here.
# ============================================================

SUPABASE_URL: str = ""          # set via env var SUPABASE_URL
SUPABASE_KEY: str = ""          # set via env var SUPABASE_SERVICE_KEY


# ============================================================
# STRATEGY FILTER
# Which strategies the optimiser will process in this run.
# ============================================================

ACTIVE_STATUSES: List[str] = ["Active"]

STRATEGY_COLUMNS: List[str] = [
    "strategy_id",
    "strategy_name",
    "deployment_type",
    "la_mapping_id",
    "capital",
    "index_name",
    "trades_type",
    "option_expiry",
    "side",
    "status",
]


# ============================================================
# LOT SIZE LOOKUP
# Instruments the engine recognises.
# ============================================================

SUPPORTED_INSTRUMENTS: List[str] = ["NIFTY", "BANKNIFTY"]


# ============================================================
# CAPITAL VOCABULARY — read this before touching anything below
#
# STRATEGY_CAPITAL  — raw value from strategies.capital column
#                     used ONCE to compute PER_UNIT_CAPITAL
#                     never used again after that
#
# PER_UNIT_CAPITAL  — STRATEGY_CAPITAL / current_lot_size
#                     the invariant anchor for all boundary
#                     calculations and ROI output
#                     does not change with lot size changes
#
# HISTORICAL_CAPITAL — PER_UNIT_CAPITAL × lot_size_on_trade_date
#                      computed per trade date in sim_daily only
#                      never used inside the simulation loop
# ============================================================


# ============================================================
# PARAMETER BOUNDARY PERCENTAGES
# Applied to PER_UNIT_CAPITAL to get point boundaries.
# Example: PER_UNIT_CAPITAL = 1000
#   SL_FLOOR_PCT   0.0075 → SL floor   = 7.5 pts
#   SL_CEILING_PCT 0.0200 → SL ceiling = 20.0 pts
# ============================================================

# SL — same for buying and selling
SL_FLOOR_PCT:            float = 0.0075   # 0.75% — min SL in points
SL_CEILING_PCT:          float = 0.0200   # 2.00% — max SL in points

# TSL activation — floor same, ceiling differs by strategy type
TSL_ACTIVATION_FLOOR_PCT:         float = 0.01   # 1% of capital
TSL_ACTIVATION_CEILING_SELLING_PCT: float = 0.10  # 10% for option selling
TSL_ACTIVATION_CEILING_BUYING_PCT:  float = 0.25  # 25% for option buying

# TSL gap — expressed as percentage of TSL activation
# (not as fixed absolute value anymore)
# gap = pct × TSL_activation → tsl_floor = (1 - pct) × TSL_activation
TSL_GAP_MIN_PCT:         float = 0.30    # gap ≥ 50% of activation
TSL_GAP_MAX_PCT:         float = 0.99    # gap ≤ 99% of activation
TSL_GAP_PCT_STEP:        float = 0.01    # 1% steps → 50 values

# Profit target — floor same, ceiling differs by strategy type
PT_FLOOR_PCT:            float = 0.01    # 1% of capital
PT_CEILING_SELLING_PCT:  float = 0.10    # 10% for option selling
PT_CEILING_BUYING_PCT:   float = 0.50    # 25% for option buying

# Legacy — kept for backward compatibility in validator
TSL_GAP_FLOOR_PCT:       float = 0.0050  # 0.50% — min absolute TSL gap


# ============================================================
# COMBO ORDERING CONSTRAINT
# Enforced by is_valid_combo() in validator.py.
# Must always hold:
#   SL < TSL_FLOOR < TSL_ACTIVATION < PROFIT_TARGET
# where TSL_FLOOR = TSL_ACTIVATION - TSL_GAP
# All values in points, all positive.
# ============================================================


# ============================================================
# UNIVERSAL EXIT TIMES
# Four allowed square-off times (24h format strings).
# Treated as a discrete parameter dimension in Optuna.
# ============================================================

UNIVERSAL_EXIT_TIMES: List[str] = [
    "14:30",
    "14:45",
    "15:00",
    "15:10",
]


# ============================================================
# OPTUNA SETTINGS
# ============================================================

OPTUNA_TRIALS:      int = 2000    # total trials per strategy per run
                                   # (tight bounds mean TPE converges
                                   #  by ~1000 trials — 2000 gives margin)
OPTUNA_TIMEOUT_SEC: int = 300     # max seconds per strategy (5 min)
OPTUNA_SAMPLER:     str = "TPE"   # TPE = Tree-structured Parzen Estimator


# ============================================================
# WALK-FORWARD VALIDATION SPLIT
# Train on older dates, validate on most recent dates.
# VALIDATION_SPLIT = 0.20 means most recent 20% of trade
# dates are held out for validation only.
# Minimum dates required to apply walk-forward at all —
# below this threshold, optimiser uses all dates for training
# with no validation split.
# ============================================================

VALIDATION_SPLIT:     float = 0   # 20% held out for validation
MIN_DATES_FOR_SPLIT:  int   = 30     # minimum trade dates to enable split


# ============================================================
# REWARD FUNCTION WEIGHTS
# Must sum to 1.0.
# Priority order: ROI > win rate > drawdown.
# ============================================================

REWARD_WEIGHT_ROI:      float = 0.70
REWARD_WEIGHT_WINRATE:  float = 0.20
REWARD_WEIGHT_DRAWDOWN: float = 0.10

assert abs(
    REWARD_WEIGHT_ROI +
    REWARD_WEIGHT_WINRATE +
    REWARD_WEIGHT_DRAWDOWN - 1.0
) < 1e-9, "Reward weights must sum to 1.0"


# ============================================================
# VALIDATION PENALTY
# If validation ROI is significantly below training ROI,
# the combo is penalised in the reward score.
# VALIDATION_PENALTY_THRESHOLD: how much worse validation
# can be before penalty kicks in (as a ROI % difference).
# VALIDATION_PENALTY_FACTOR: multiplier applied to reward
# score when penalty triggers (< 1.0 reduces the score).
# ============================================================

VALIDATION_PENALTY_THRESHOLD: float = 5.0   # pct points
VALIDATION_PENALTY_FACTOR:    float = 0.80  # 20% penalty on reward score


# ============================================================
# PARAMETER STEP SIZES
# Optuna samples continuously but we round to these steps
# to keep combos practical and avoid over-precision.
# All in points.
# ============================================================

SL_STEP_PTS:             float = 0.5    # SL step unchanged
TSL_ACTIVATION_STEP_PTS: float = 2.0    # coarser step for activation
PT_STEP_PTS:             float = 2.0    # coarser step for profit target
# TSL_GAP uses TSL_GAP_PCT_STEP (0.01) above — sampled as
# percentage of activation, then converted to absolute pts


# ============================================================
# OUTPUT SETTINGS
# ============================================================

# How many decimal places to round points values in output
POINTS_PRECISION: int = 1

# How many decimal places to round INR values in output
INR_PRECISION: int = 2

# How many decimal places to round percentage values in output
PCT_PRECISION: int = 2


# ============================================================
# RUNTIME BOUNDARY CALCULATOR
# Called once per strategy at startup.
# Returns all point boundaries derived from PER_UNIT_CAPITAL.
# This is the only function that touches the boundary PCTs.
# ============================================================

@dataclass
class StrategyBoundaries:
    """
    All search space boundaries in points for one strategy.
    Computed once from PER_UNIT_CAPITAL at engine startup.

    is_buying determines whether to use buying (1-25%) or
    selling (1-10%) ceilings for TSL activation and PT.

    Classification rule:
        is_buying = (trades_type == 'Option Buying')
                    or (side == 'Buy')
    """
    per_unit_capital:        float
    is_buying:               bool = False

    # SL bounds
    sl_floor_pts:            float = field(init=False)
    sl_ceiling_pts:          float = field(init=False)

    # TSL activation bounds
    tsl_activation_floor_pts:   float = field(init=False)
    tsl_activation_ceiling_pts: float = field(init=False)

    # TSL gap — absolute floor (legacy) for validator
    tsl_gap_floor_pts:       float = field(init=False)

    # Profit target bounds
    pt_floor_pts:            float = field(init=False)
    pt_ceiling_pts:          float = field(init=False)

    def __post_init__(self):
        puc = self.per_unit_capital

        # SL — same for both types
        self.sl_floor_pts = round(
            puc * SL_FLOOR_PCT, POINTS_PRECISION
        )
        self.sl_ceiling_pts = round(
            puc * SL_CEILING_PCT, POINTS_PRECISION
        )

        # TSL activation
        self.tsl_activation_floor_pts = round(
            puc * TSL_ACTIVATION_FLOOR_PCT, POINTS_PRECISION
        )
        if self.is_buying:
            self.tsl_activation_ceiling_pts = round(
                puc * TSL_ACTIVATION_CEILING_BUYING_PCT,
                POINTS_PRECISION
            )
        else:
            self.tsl_activation_ceiling_pts = round(
                puc * TSL_ACTIVATION_CEILING_SELLING_PCT,
                POINTS_PRECISION
            )

        # TSL gap — legacy absolute floor for validator
        self.tsl_gap_floor_pts = round(
            puc * TSL_GAP_FLOOR_PCT, POINTS_PRECISION
        )

        # Profit target
        self.pt_floor_pts = round(
            puc * PT_FLOOR_PCT, POINTS_PRECISION
        )
        if self.is_buying:
            self.pt_ceiling_pts = round(
                puc * PT_CEILING_BUYING_PCT,
                POINTS_PRECISION
            )
        else:
            self.pt_ceiling_pts = round(
                puc * PT_CEILING_SELLING_PCT,
                POINTS_PRECISION
            )

    def describe(self) -> str:
        kind = "BUYING" if self.is_buying else "SELLING"
        return (
            f"Boundaries ({kind}, "
            f"PER_UNIT_CAPITAL={self.per_unit_capital}):\n"
            f"  SL:             {self.sl_floor_pts}"
            f" – {self.sl_ceiling_pts} pts\n"
            f"  TSL activation: {self.tsl_activation_floor_pts}"
            f" – {self.tsl_activation_ceiling_pts} pts\n"
            f"  TSL gap:        {int(TSL_GAP_MIN_PCT*100)}%"
            f" – {int(TSL_GAP_MAX_PCT*100)}% of activation\n"
            f"  PT:             {self.pt_floor_pts}"
            f" – {self.pt_ceiling_pts} pts\n"
        )
