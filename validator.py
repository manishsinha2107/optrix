# validator.py
# ============================================================
# Enforces all combo validity rules before any trial is
# scored by the simulator.
#
# Two responsibilities:
#   1. is_valid_combo() — the hard gate that Optuna calls
#      before every trial. Invalid combos are rejected
#      immediately without running the simulation.
#   2. Detailed validation report for logging/debugging.
#
# All values passed in and checked here are in POINTS.
# No INR, no HISTORICAL_CAPITAL, no STRATEGY_CAPITAL.
#
# Rules enforced (all in points):
#   R1. SL >= sl_floor_pts
#   R2. SL <= sl_ceiling_pts
#   R3. tsl_activation >= tsl_activation_floor_pts
#   R3b. tsl_activation <= tsl_activation_ceiling_pts
#   R4. tsl_gap >= tsl_gap_floor_pts
#   R4b. tsl_gap within 50%–99% of tsl_activation
#   R5. tsl_floor = tsl_activation - tsl_gap
#       tsl_floor > SL   (ordering constraint)
#   R6. tsl_activation > tsl_floor  (always true if R4 holds
#       but checked explicitly for safety)
#   R7. profit_target > tsl_activation  (ordering constraint)
#   R7b. profit_target <= pt_ceiling_pts
#   R8. universal_exit_time in allowed set
#
# Ordering summary:
#   SL < tsl_floor < tsl_activation < profit_target
# ============================================================

import logging
from dataclasses import dataclass

from config import (
    UNIVERSAL_EXIT_TIMES,
    POINTS_PRECISION,
    TSL_GAP_MIN_PCT,
    TSL_GAP_MAX_PCT,
    StrategyBoundaries,
)

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION RESULT
# ============================================================

@dataclass
class ValidationResult:
    """
    Result of is_valid_combo() with full detail
    for logging and debugging.
    """
    is_valid:        bool
    failed_rule:     str   = ""    # empty if valid
    detail:          str   = ""    # human readable reason

    def __bool__(self) -> bool:
        return self.is_valid


# ============================================================
# CORE VALIDATOR
# ============================================================

def validate_combo(
    sl_pts:               float,
    tsl_activation_pts:   float,
    tsl_gap_pts:          float,
    pt_pts:               float,
    universal_exit_time:  str,
    boundaries:           StrategyBoundaries,
) -> ValidationResult:
    """
    Validate one combo against all rules.
    Returns a ValidationResult — check .is_valid before
    running the simulator.

    All numeric inputs must be in points.
    All comparisons are in points.

    Args:
        sl_pts:              stop loss in points (positive)
        tsl_activation_pts:  TSL activation level in points
        tsl_gap_pts:         TSL gap in points
                             (tsl_floor = activation - gap)
        pt_pts:              profit target in points
        universal_exit_time: one of UNIVERSAL_EXIT_TIMES
        boundaries:          StrategyBoundaries instance for
                             this strategy

    Returns:
        ValidationResult with is_valid, failed_rule, detail
    """

    # --------------------------------------------------------
    # R1. SL >= floor
    # --------------------------------------------------------
    if sl_pts < boundaries.sl_floor_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R1_SL_BELOW_FLOOR",
            detail=(
                f"sl_pts={sl_pts} < "
                f"sl_floor={boundaries.sl_floor_pts}. "
                f"SL is too tight for intraday options."
            ),
        )

    # --------------------------------------------------------
    # R2. SL <= ceiling
    # --------------------------------------------------------
    if sl_pts > boundaries.sl_ceiling_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R2_SL_ABOVE_CEILING",
            detail=(
                f"sl_pts={sl_pts} > "
                f"sl_ceiling={boundaries.sl_ceiling_pts}. "
                f"SL exposes too much capital."
            ),
        )

    # --------------------------------------------------------
    # R3. TSL activation >= floor
    # --------------------------------------------------------
    if tsl_activation_pts < boundaries.tsl_activation_floor_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R3_TSL_ACTIVATION_BELOW_FLOOR",
            detail=(
                f"tsl_activation_pts={tsl_activation_pts} < "
                f"activation_floor="
                f"{boundaries.tsl_activation_floor_pts}. "
                f"TSL would activate on noise."
            ),
        )

    # --------------------------------------------------------
    # R4. TSL gap >= floor
    # --------------------------------------------------------
    if tsl_gap_pts < boundaries.tsl_gap_floor_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R4_TSL_GAP_BELOW_FLOOR",
            detail=(
                f"tsl_gap_pts={tsl_gap_pts} < "
                f"gap_floor={boundaries.tsl_gap_floor_pts}. "
                f"Gap too small — meaningless retracement "
                f"would trigger TSL exit."
            ),
        )

    # --------------------------------------------------------
    # R3b. TSL activation <= ceiling
    # --------------------------------------------------------
    if tsl_activation_pts > boundaries.tsl_activation_ceiling_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R3b_TSL_ACTIVATION_ABOVE_CEILING",
            detail=(
                f"tsl_activation_pts={tsl_activation_pts} > "
                f"ceiling="
                f"{boundaries.tsl_activation_ceiling_pts}. "
                f"TSL activation exceeds strategy type limit."
            ),
        )

    # --------------------------------------------------------
    # R4b. TSL gap within 50%–99% of TSL activation
    # --------------------------------------------------------
    gap_pct = tsl_gap_pts / tsl_activation_pts if tsl_activation_pts > 0 else 0
    if gap_pct < TSL_GAP_MIN_PCT - 1e-9:
        return ValidationResult(
            is_valid=False,
            failed_rule="R4b_TSL_GAP_BELOW_MIN_PCT",
            detail=(
                f"tsl_gap={tsl_gap_pts} is "
                f"{gap_pct*100:.1f}% of activation "
                f"({tsl_activation_pts}). "
                f"Must be >= {TSL_GAP_MIN_PCT*100:.0f}%."
            ),
        )
    if gap_pct > TSL_GAP_MAX_PCT + 1e-9:
        return ValidationResult(
            is_valid=False,
            failed_rule="R4b_TSL_GAP_ABOVE_MAX_PCT",
            detail=(
                f"tsl_gap={tsl_gap_pts} is "
                f"{gap_pct*100:.1f}% of activation "
                f"({tsl_activation_pts}). "
                f"Must be <= {TSL_GAP_MAX_PCT*100:.0f}%."
            ),
        )

    # --------------------------------------------------------
    # Derived value: tsl_floor
    # --------------------------------------------------------
    tsl_floor_pts = round(
        tsl_activation_pts - tsl_gap_pts,
        POINTS_PRECISION
    )

    # --------------------------------------------------------
    # R5. Ordering: SL < tsl_floor
    # If gap is so large that tsl_floor <= SL, the TSL
    # would never protect more than the SL already does.
    # --------------------------------------------------------
    if tsl_floor_pts <= sl_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R5_TSL_FLOOR_NOT_ABOVE_SL",
            detail=(
                f"tsl_floor_pts={tsl_floor_pts} <= "
                f"sl_pts={sl_pts}. "
                f"Ordering violated: SL must be < TSL floor. "
                f"Gap too large relative to activation."
            ),
        )

    # --------------------------------------------------------
    # R6. Ordering: tsl_floor < tsl_activation
    # Mathematically guaranteed if gap > 0 and R4 passed,
    # but checked explicitly to catch floating point edge
    # cases.
    # --------------------------------------------------------
    if tsl_floor_pts >= tsl_activation_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R6_TSL_FLOOR_NOT_BELOW_ACTIVATION",
            detail=(
                f"tsl_floor_pts={tsl_floor_pts} >= "
                f"tsl_activation_pts={tsl_activation_pts}. "
                f"Ordering violated: TSL floor must be "
                f"< TSL activation."
            ),
        )

    # --------------------------------------------------------
    # R7. Ordering: tsl_activation < profit_target
    # --------------------------------------------------------
    if pt_pts <= tsl_activation_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R7_PT_NOT_ABOVE_TSL_ACTIVATION",
            detail=(
                f"pt_pts={pt_pts} <= "
                f"tsl_activation_pts={tsl_activation_pts}. "
                f"Ordering violated: profit target must be "
                f"> TSL activation."
            ),
        )

    # --------------------------------------------------------
    # R7b. Profit target <= ceiling
    # --------------------------------------------------------
    if pt_pts > boundaries.pt_ceiling_pts:
        return ValidationResult(
            is_valid=False,
            failed_rule="R7b_PT_ABOVE_CEILING",
            detail=(
                f"pt_pts={pt_pts} > "
                f"pt_ceiling={boundaries.pt_ceiling_pts}. "
                f"Profit target exceeds strategy type limit."
            ),
        )

    # --------------------------------------------------------
    # R8. Universal exit time in allowed set
    # --------------------------------------------------------
    if universal_exit_time not in UNIVERSAL_EXIT_TIMES:
        return ValidationResult(
            is_valid=False,
            failed_rule="R8_INVALID_EXIT_TIME",
            detail=(
                f"universal_exit_time='{universal_exit_time}' "
                f"not in allowed set: {UNIVERSAL_EXIT_TIMES}."
            ),
        )

    # --------------------------------------------------------
    # All rules passed
    # --------------------------------------------------------
    return ValidationResult(is_valid=True)


def is_valid_combo(
    sl_pts:              float,
    tsl_activation_pts:  float,
    tsl_gap_pts:         float,
    pt_pts:              float,
    universal_exit_time: str,
    boundaries:          StrategyBoundaries,
) -> bool:
    """
    Thin wrapper around validate_combo() for use inside
    the Optuna objective function where only a bool is needed.

    Called millions of times — kept as fast as possible.
    No logging at this level to avoid I/O overhead in the
    hot loop. Use validate_combo() directly when you need
    the failure reason.
    """
    return validate_combo(
        sl_pts=sl_pts,
        tsl_activation_pts=tsl_activation_pts,
        tsl_gap_pts=tsl_gap_pts,
        pt_pts=pt_pts,
        universal_exit_time=universal_exit_time,
        boundaries=boundaries,
    ).is_valid


# ============================================================
# COMBO LABEL BUILDER
# Produces the human readable UI column header string.
# Placed here because it depends on the same derived values
# (tsl_floor) that the validator computes.
# ============================================================

def build_combo_label(
    sl_pts:              float,
    tsl_activation_pts:  float,
    tsl_gap_pts:         float,
    pt_pts:              float,
    universal_exit_time: str,
    current_lot_size:    int,
) -> str:
    """
    Build the human readable combo label for UI display.

    Format:
        "SL -975 | T 1300 | TSL 780 | PT 2925 | Exit 15:00"

    Where:
        SL  = sl_pts × current_lot_size (negative, shown as loss)
        T   = tsl_activation_pts × current_lot_size
        TSL = tsl_floor_pts × current_lot_size
              (tsl_floor = activation - gap, the trail level
               shown to user — not the internal gap)
        PT  = pt_pts × current_lot_size

    All INR values rounded to nearest integer for readability.

    Args:
        sl_pts:             stop loss in points
        tsl_activation_pts: TSL activation in points
        tsl_gap_pts:        TSL gap in points
        pt_pts:             profit target in points
        universal_exit_time: e.g. "15:00"
        current_lot_size:   current lot size for INR conversion

    Returns:
        label string e.g.
        "SL -975 | T 1300 | TSL 780 | PT 2925 | Exit 15:00"
    """
    tsl_floor_pts = tsl_activation_pts - tsl_gap_pts

    sl_inr   = -abs(round(sl_pts * current_lot_size))
    t_inr    = round(tsl_activation_pts * current_lot_size)
    tsl_inr  = round(tsl_floor_pts * current_lot_size)
    pt_inr   = round(pt_pts * current_lot_size)

    return (
        f"SL {sl_inr} | "
        f"T {t_inr} | "
        f"TSL {tsl_inr} | "
        f"PT {pt_inr} | "
        f"Exit {universal_exit_time}"
    )


# ============================================================
# DIAGNOSTIC HELPER
# Used during development and logging to inspect why a
# specific combo failed validation. Not called in hot loop.
# ============================================================

def explain_combo(
    sl_pts:              float,
    tsl_activation_pts:  float,
    tsl_gap_pts:         float,
    pt_pts:              float,
    universal_exit_time: str,
    boundaries:          StrategyBoundaries,
) -> str:
    """
    Return a full human readable explanation of a combo
    including all derived values and which rules pass/fail.
    Useful for debugging config boundaries or understanding
    why a specific combo was rejected.

    Example output:
        Combo explanation:
          sl_pts            = 7.5
          tsl_activation    = 20.0
          tsl_gap           = 8.0
          tsl_floor         = 12.0  (activation - gap)
          pt_pts            = 45.0
          universal_exit    = 15:00
          Ordering check:   7.5 < 12.0 < 20.0 < 45.0  ✓
          Boundaries:
            SL floor        = 7.5   PASS
            SL ceiling      = 20.0  PASS
            TSL act floor   = 10.0  PASS
            TSL gap floor   = 5.0   PASS
          Result: VALID
    """
    tsl_floor = round(tsl_activation_pts - tsl_gap_pts,
                      POINTS_PRECISION)
    result = validate_combo(
        sl_pts=sl_pts,
        tsl_activation_pts=tsl_activation_pts,
        tsl_gap_pts=tsl_gap_pts,
        pt_pts=pt_pts,
        universal_exit_time=universal_exit_time,
        boundaries=boundaries,
    )

    r1 = "PASS" if sl_pts >= boundaries.sl_floor_pts       else "FAIL"
    r2 = "PASS" if sl_pts <= boundaries.sl_ceiling_pts     else "FAIL"
    r3 = "PASS" if tsl_activation_pts >= \
         boundaries.tsl_activation_floor_pts               else "FAIL"
    r3b = "PASS" if tsl_activation_pts <= \
          boundaries.tsl_activation_ceiling_pts             else "FAIL"
    r4 = "PASS" if tsl_gap_pts >= \
         boundaries.tsl_gap_floor_pts                      else "FAIL"
    gap_pct = tsl_gap_pts / tsl_activation_pts if tsl_activation_pts > 0 else 0
    r4b = "PASS" if TSL_GAP_MIN_PCT - 1e-9 <= gap_pct <= TSL_GAP_MAX_PCT + 1e-9 else "FAIL"
    r5 = "PASS" if tsl_floor > sl_pts                      else "FAIL"
    r6 = "PASS" if tsl_floor < tsl_activation_pts          else "FAIL"
    r7 = "PASS" if pt_pts > tsl_activation_pts             else "FAIL"
    r7b = "PASS" if pt_pts <= boundaries.pt_ceiling_pts    else "FAIL"
    r8 = "PASS" if universal_exit_time \
         in UNIVERSAL_EXIT_TIMES                           else "FAIL"

    ordering = (
        f"{sl_pts} < {tsl_floor} < "
        f"{tsl_activation_pts} < {pt_pts}"
    )
    valid_str = "VALID" if result.is_valid else \
        f"INVALID ({result.failed_rule}: {result.detail})"

    return (
        f"Combo explanation:\n"
        f"  sl_pts            = {sl_pts}\n"
        f"  tsl_activation    = {tsl_activation_pts}\n"
        f"  tsl_gap           = {tsl_gap_pts}"
        f"  ({gap_pct*100:.1f}% of activation)\n"
        f"  tsl_floor         = {tsl_floor}  "
        f"(activation - gap)\n"
        f"  pt_pts            = {pt_pts}\n"
        f"  universal_exit    = {universal_exit_time}\n"
        f"  Ordering:         {ordering}\n"
        f"  R1  SL >= floor      {boundaries.sl_floor_pts}"
        f"            {r1}\n"
        f"  R2  SL <= ceiling    {boundaries.sl_ceiling_pts}"
        f"           {r2}\n"
        f"  R3  TSL act >= floor "
        f"{boundaries.tsl_activation_floor_pts}"
        f"           {r3}\n"
        f"  R3b TSL act <= ceil  "
        f"{boundaries.tsl_activation_ceiling_pts}"
        f"          {r3b}\n"
        f"  R4  TSL gap >= floor {boundaries.tsl_gap_floor_pts}"
        f"            {r4}\n"
        f"  R4b TSL gap 50-99%             {r4b}\n"
        f"  R5  SL < TSL floor             {r5}\n"
        f"  R6  floor < activation         {r6}\n"
        f"  R7  activation < PT            {r7}\n"
        f"  R7b PT <= ceiling    "
        f"{boundaries.pt_ceiling_pts}"
        f"          {r7b}\n"
        f"  R8  exit time valid            {r8}\n"
        f"  Result: {valid_str}"
    )
