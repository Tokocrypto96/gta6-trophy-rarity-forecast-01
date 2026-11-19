"""GTA VI trophy earn-rate forecast (make.fun gta6-trophy-rarity-forecast-01).

For each of the game's 55 PS5 trophies, forecast the Exophase earn rate
(percent of tracked players) at the case's target instant -- the final
case targets 2026-11-26T00:00:00Z, one week after launch -- as five
quantiles (5%, 25%, 50%, 75%, 95%).

The forecast is built per trophy, offline, with the standard library
only, and is fully deterministic (no randomness, no dict-order
dependence, sorted case files):

* Rates live in logit space, L = ln(p / (1 - p)), with the percent
  clamped to [0.01, 99.99] first, exactly as the house baseline does.
  On that scale a move from 1% to 2% counts about as much as a move
  from 33% to 50%, which is also how the contest scores (pinball loss
  on the logit scale).
* The clock is log-time, x = ln(1 + hours since the launch reference
  2026-11-19T00:00:00Z). Player growth and player progress both
  decelerate after launch, so a straight line in log-time bends the
  extrapolation down instead of letting it run away.
* Trend: a recency-weighted least-squares line of L against x over
  the last 24 hours of reads, weights decaying with an 8-hour
  half-life, so the fit tracks the current direction rather than the
  launch-night average. The median is the last observed logit plus
  the line's move to the target, damped by 0.7 -- the future slows
  down further than the fit window shows.
* Flat history: with fewer than 3 reads, or every rate identical,
  there is no trend to fit, so the median persists the last rate.
  But if the tracked-player count is still under 10,000, the page
  is pre-refresh: Exophase refreshes a game's counts on its own
  schedule, and when it does, the player count jumps and every rate
  except a 100% one falls (launch night: 638 -> 3,868 players at the
  04:30Z refresh, every rate lower). So a flat, early history gets a
  dilution prior: median 0.4x the last rate for trophies below 99%.
  A real (non-flat) history overrides this prior.
* Spread, in logit units: the house baseline's random walk,
  0.5 * sqrt(horizon in days), combined in quadrature with the
  trend fit's residual and 0.3 * |damped logit move| (a large
  extrapolated move is an uncertain one), capped at 3.0. The
  quantiles are median_logit + z_tau * spread, mapped back to
  percent and rounded to 4 decimals, half up, as the house baseline
  rounds; a final pass enforces non-decreasing order after rounding.

The grader runs `python3 -P -s /bot/main.py` (python:3.13-slim,
stdlib only, no network), one case per run, reading every <id>.in
in the input directory and writing <id>.out to the output directory:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in
for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P keeps this file's directory off sys.path; restore it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at those levels.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Launch reference for the log-time clock.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

RW_SIGMA = 0.5          # random-walk spread, logit units per sqrt(day)
DAMP = 0.7              # damping on the extrapolated logit move
HALF_LIFE_H = 8.0       # recency-weight half-life for the trend fit
WINDOW_H = 24.0         # trend fit window, hours back from last read
DILUTION = 0.4          # flat-early median as a fraction of last rate
DILUTION_PLAYERS = 10000  # dilution prior only below this player count
SPREAD_CAP = 3.0        # spread cap, logit units
LO = 0.01               # rate clamp bounds, percent
HI = 99.99


def clamp_pct(rate):
    """Clamp a percent into the range the logit is defined on."""
    return HI if rate > HI else LO if rate < LO else rate


def to_logit(rate):
    p = clamp_pct(rate) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    """Logit back to a percent, numerically safe for large |lg|."""
    if lg >= 0.0:
        p = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        p = e / (1.0 + e)
    return 100.0 * p


def round4(value):
    """Round to 4 decimals, half up (JavaScript Math.round semantics).

    Whole numbers come back as int, so json writes 100, not 100.0.
    """
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_clock(moment):
    """ln(1 + hours since launch); reads before launch count as hour 0."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log1p(hours if hours > 0.0 else 0.0)


def weighted_fit(xs, ys, ws):
    """Weighted least squares for y = intercept + slope * x.

    Returns (intercept, slope, rmse), or None when the weights sum to
    nothing. A zero-variance x collapses to a flat line at the
    weighted mean, with zero residual.
    """
    total_w = sum(ws)
    if total_w <= 0.0:
        return None
    x_bar = sum(w * x for x, w in zip(xs, ws)) / total_w
    y_bar = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (y_bar, 0.0, 0.0)
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    sq = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    return (intercept, slope, math.sqrt(sq / total_w))


def make_quantiles(median_logit, spread):
    """Five percent quantiles around a median logit, non-decreasing."""
    qs = []
    for z in Z_SCORES:
        q = from_logit(median_logit + z * spread)
        qs.append(round4(min(100.0, max(0.0, q))))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def predict_trophy(reads, players_last, horizon_days, target):
    """Forecast one trophy.

    reads: [(datetime, rate or None)] oldest first.
    players_last: players_tracked of the case's last read (or None).
    horizon_days: days from the last read to the target (at least the
    case's stated horizon).
    Returns the five quantiles, in percent.
    """
    observed = [(t, r) for (t, r) in reads if r is not None]
    walk = RW_SIGMA * math.sqrt(horizon_days if horizon_days > 0.0 else 0.0)
    if not observed:
        # No rate ever seen: centre on 1% with a wide band.
        return make_quantiles(to_logit(1.0), min(walk + 0.5, SPREAD_CAP))

    last_t, last_rate = observed[-1]
    last_logit = to_logit(last_rate)
    distinct_rates = {r for (_t, r) in observed}

    if len(observed) < 3 or len(distinct_rates) == 1:
        # No trend to fit: persist, with the dilution prior when the
        # history is flat and the page is still pre-refresh.
        median_rate = last_rate
        widened = walk
        if (
            players_last
            and players_last < DILUTION_PLAYERS
            and last_rate < 99.0
        ):
            median_rate = clamp_pct(last_rate * DILUTION)
            # The prior is a guess, so widen the band for it.
            widened = math.sqrt(walk * walk + 0.6 * 0.6)
        return make_quantiles(to_logit(median_rate), min(widened, SPREAD_CAP))

    # Trend path: last 24 h of reads (falling back to all of them if
    # that window holds too few), logit vs log-time, recency-weighted.
    cutoff = last_t.timestamp() - WINDOW_H * 3600.0
    window = [(t, r) for (t, r) in observed if t.timestamp() >= cutoff]
    if len(window) < 3:
        window = observed
    xs = [log_clock(t) for (t, _r) in window]
    ys = [to_logit(r) for (_t, r) in window]
    ws = [
        0.5 ** ((last_t - t).total_seconds() / 3600.0 / HALF_LIFE_H)
        for (t, _r) in window
    ]
    fit = weighted_fit(xs, ys, ws)
    if fit is None:
        return make_quantiles(last_logit, min(walk, SPREAD_CAP))
    _intercept, slope, rmse = fit

    # Extrapolate from the last observed logit, damped, and keep the
    # median inside the logit range the clamp allows.
    ahead = log_clock(target) - log_clock(last_t)
    move = slope * (ahead if ahead > 0.0 else 0.0) * DAMP
    median_logit = last_logit + move
    median_logit = max(to_logit(LO), min(to_logit(HI), median_logit))

    # Spread: random walk + fit residual + trend uncertainty.
    spread = math.sqrt(walk ** 2 + rmse ** 2 + (0.3 * abs(move)) ** 2)
    return make_quantiles(median_logit, min(spread, SPREAD_CAP))


def forecast_case(case, label):
    if list(case["levels"]) != list(LEVELS):
        raise ValueError(f"{label}: levels {case['levels']} != {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{label}: empty history")
    target = parse_ts(case["target"])
    last_read = case["history"][-1]
    last_at = parse_ts(last_read["at"])
    horizon_days = (target - last_at).total_seconds() / 86400.0
    stated = case.get("horizon_minutes")
    if stated is not None:
        horizon_days = max(horizon_days, stated / 1440.0)
    horizon_days = max(0.0, horizon_days)

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        reads = []
        for read in case["history"]:
            rates = read.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            reads.append((parse_ts(read["at"]), rate))
        rows.append(
            {
                "id": trophy["id"],
                "q": predict_trophy(
                    reads, last_read.get("players_tracked"), horizon_days, target
                ),
            }
        )
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
