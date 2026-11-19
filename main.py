"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
at the case target (2026-11-26T00:00:00Z) as five quantiles (5/25/50/75/95%).

Method (per trophy, standard library only, offline, deterministic):
  1. Take the case history (reads of Exophase's trophy page, oldest first).
     Each read gives the trophy's earn rate (percent of tracked players) and
     the players-tracked count.
  2. Work in logit space: L = ln(p / (1 - p)), p = rate / 100, rate clamped
     to [0.01, 99.99] before the logit (as the house baseline does).
  3. Fit a straight line of L against LOG-TIME, x = ln(1 + hours since the
     launch reference 2026-11-19T00:00:00Z), over the case's last 24 hours
     of reads, by recency-weighted least squares (exponential weights,
     half-life 8 h, so the fit follows the current direction, not the
     launch-night average). Log-time is the right clock here: both the
     player base's growth and players' progress decelerate, so a line in
     log-time extrapolates a flattening curve, not a runaway one.
  4. Read the line at the target's log-time for the median. Damp the move
     from the last observed logit by 0.7: the future decelerates further
     than the fit's window shows, and an undamped line overshoots.
     Clamp the median rate to [0.01, 99.99].
  5. Flat history (< 3 reads, or every rate identical): no trend to fit.
     Fall back to the last rate (persistence). If the tracked-player count
     is still small (< 10,000 -- launch night, before Exophase's first
     refresh), apply a dilution prior: Exophase refreshes a game's counts
     on its own schedule, and when it does the tracked-player count jumps
     and every rate but a 100% one falls (RULES.txt: 638 -> 3,868 players
     at the 04:30Z refresh on launch night, every rate lower). So for a
     flat, early history, predict the median at 0.4x the last rate for
     trophies below 99% (the 100% trophy stays ~99%). This only fires
     when there is no trend signal at all; a real history overrides it.
  6. Spread (logit units): random walk 0.5 * sqrt(horizon_days) (the house
     baseline's term) combined in quadrature with the fit residual (how
     well the line fits the last 24 h) and a trend-uncertainty term
     0.3 * |damped logit move| (a big extrapolated move is an uncertain
     one). Capped at 3.0 logit units. The quantiles are the median logit
     plus z_tau * spread, mapped back to percent, rounded to 4 decimals
     (half up, as the house baseline rounds).

The house baseline (persistence + random-walk spread) scores about 1.51
pinball loss on the practice cases, because the refresh takes every rate
outside its narrow 90-minute band. This model beats it by (a) following
the trend where the history has one, instead of persisting a stale rate,
and (b) anticipating the refresh dilution when the history is flat and
early, instead of persisting a pre-refresh rate.

Grader: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), python:3.13-slim,
stdlib only, no network. Reads every <id>.in in the input directory and
writes <id>.out to the output directory:
  {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
one row per trophy, each q five finite numbers 0..100, non-decreasing.
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch reference for the log-time clock (the house curve baseline's).
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per square-root day (house baseline).
SIGMA_RW = 0.5
# Damping on the extrapolated logit move (0.7 = trust the trend, but not fully).
DAMP = 0.7
# Recency weighting half-life for the trend fit, in hours.
HALF_LIFE_H = 8.0
# Fit window: last 24 hours of reads.
WINDOW_H = 24.0
# Dilution prior: when the history is flat and early (few tracked players),
# the median is this fraction of the last rate (see docstring, point 5).
DILUTION_FACTOR = 0.4
DILUTION_MAX_PLAYERS = 10000
# Spread cap, logit units.
SPREAD_CAP = 3.0


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(t: datetime) -> float:
    """ln(1 + hours since launch). Negative hours clamp to 0."""
    hours = (t - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, rmse)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for _, y, w in zip(xs, ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / sw)
    return (a, b, rmse)


def forecast_trophy(series, horizon_days, target_dt):
    """series: list of (datetime, rate_or_None, players_tracked).
    Returns the five quantiles (percent) for this trophy."""
    # Keep only reads with a rate, oldest first.
    pts = [(t, r) for (t, r, _n) in series if r is not None]
    if not pts:
        # No data at all: centre on 1% with a wide spread.
        med_logit = logit(1.0)
        spread = SIGMA_RW * math.sqrt(max(horizon_days, 0.0)) + 0.5
        return quantiles(med_logit, min(spread, SPREAD_CAP))
    last_t, last_rate = pts[-1]
    last_logit = logit(last_rate)
    n_players = series[-1][2] if series[-1][2] is not None else 0

    # Flat history (or too few reads): no trend to fit.
    rates = [r for (_t, r) in pts]
    if len(pts) < 3 or len(set(rates)) == 1:
        med_rate = last_rate
        if n_players and n_players < DILUTION_MAX_PLAYERS and last_rate < 99.0:
            # Early, flat, pre-refresh: anticipate the dilution drop.
            med_rate = clamp_rate(last_rate * DILUTION_FACTOR)
        med_logit = logit(med_rate)
        spread = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
        # The dilution prior is a guess: widen the spread for it.
        if med_rate != last_rate:
            spread = math.sqrt(spread ** 2 + 0.6 ** 2)
        return quantiles(med_logit, min(spread, SPREAD_CAP))

    # Trend fit: last 24 h of reads, logit vs log-time, recency-weighted.
    cutoff = last_t.timestamp() - WINDOW_H * 3600.0
    win = [(t, r) for (t, r) in pts if t.timestamp() >= cutoff]
    if len(win) < 3:
        win = pts
    xs = [log_time(t) for (t, _r) in win]
    ys = [logit(r) for (_t, r) in win]
    # Exponential recency weights, half-life 8 h, anchored at the last read.
    ws = [0.5 ** ((last_t - t).total_seconds() / 3600.0 / HALF_LIFE_H) for (t, _r) in win]
    fit = weighted_line(xs, ys, ws)
    if fit is None:
        return quantiles(last_logit, SIGMA_RW * math.sqrt(max(horizon_days, 0.0)))
    _a, b, rmse = fit

    # Extrapolate from the last observed logit, damped.
    x_target = log_time(target_dt)
    x_last = log_time(last_t)
    move = b * max(0.0, x_target - x_last) * DAMP
    med_logit = last_logit + move
    # Clamp the median to the valid logit range.
    med_logit = max(logit(0.01), min(logit(99.99), med_logit))

    # Spread: random walk + fit residual + trend uncertainty, in quadrature.
    rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    trend_unc = 0.3 * abs(move)
    spread = math.sqrt(rw ** 2 + rmse ** 2 + trend_unc ** 2)
    return quantiles(med_logit, min(spread, SPREAD_CAP))


def quantiles(med_logit: float, spread: float):
    out = []
    for z in Z:
        q = 100.0 * sigmoid(med_logit + z * spread)
        q = min(100.0, max(0.0, q))
        out.append(round4(q))
    # Guarantee non-decreasing after rounding.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target_dt = parse_time(case["target"])
    last_at = parse_time(case["history"][-1]["at"])
    horizon_days = max(0.0, (target_dt - last_at).total_seconds() / 86400.0)
    # Also honour the case's stated horizon if it is longer (deadline -> target).
    stated = case.get("horizon_minutes")
    if stated is not None:
        horizon_days = max(horizon_days, stated / 1440.0)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = []
        for read in case["history"]:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            series.append((parse_time(read["at"]), rate, read.get("players_tracked")))
        rows.append({"id": trophy["id"], "q": forecast_trophy(series, horizon_days, target_dt)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
