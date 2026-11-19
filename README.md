# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tokocrypto96/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tokocrypto96/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the [make.fun](https://make.fun) bounty
**gta6-trophy-rarity-forecast-01**: forecast, for every trophy of
*Grand Theft Auto VI* on PlayStation 5, the percent of the players
Exophase tracks for the game who will have earned it at
**2026-11-26T00:00:00Z** (a week after launch), as five quantiles
(5% / 25% / 50% / 75% / 95%) per trophy. Scored by mean pinball loss
on the logit scale. Prize $250 (1st $50, 2nd $35, 3rd $25,
4th–10th $10, 11th–20th $7).

## The data

The game launched 19 November 2026 on PS5 and Xbox Series X|S.
Exophase tracks every player who syncs a profile and prints each
trophy's rarity — the percent of tracked players who have earned it —
on one public page:

https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/

The trophy list (55 trophies: **1 Platinum, 2 Gold, 6 Silver,
46 Bronze**, 28 hidden) is in [`data/trophies.json`](data/trophies.json),
taken from the house recorder's practice case (2026-11-19) and
cross-checked against the press-reported list (Push Square citing
Exophase; GamesRadar+ / Press Start / PC Gamer, 18 Nov 2026).
The `points` field in that file is Exophase EXP (Platinum 300,
Gold 90, Silver 30, Bronze 15); on Xbox the same 55 achievements
are worth 1,000 Gamerscore in total.

Key facts the model uses (documented in the competition rules):
- Exophase refreshes a game's counts on its own schedule. On launch
  night the page was unchanged (638 players tracked, every rate the
  same) from 2026-11-18T23:48Z through 2026-11-19T04:00Z, then
  refreshed at 04:30Z: **3,868 players tracked, and every rate but
  the one at 100% lower than before**. A refresh dilutes the rates:
  new players join faster than they earn trophies.
- The house baseline (persist the last rate, random-walk spread)
  scores about 1.51 pinball loss on the practice cases, because the
  refresh takes every rate outside its narrow 90-minute band.

## The method

`main.py` (Python 3.13, standard library only, offline,
deterministic) reads one case file at a time and writes one forecast
file. Per trophy:

1. **Logit space.** Work with `L = ln(p / (1 - p))`, `p = rate / 100`,
   rate clamped to [0.01, 99.99] before the logit.
2. **Log-time trend.** Fit a straight line of `L` against
   `x = ln(1 + hours since 2026-11-19T00:00:00Z)` over the case's
   last 24 hours of reads, by recency-weighted least squares
   (exponential weights, 8-hour half-life). Log-time is the right
   clock: both the player base's growth and players' progress
   decelerate, so a line in log-time extrapolates a flattening
   curve, not a runaway one.
3. **Damped extrapolation.** Read the line at the target, but damp
   the move from the last observed logit by 0.7 — the future
   decelerates further than the fit window shows.
4. **Flat history fallback.** With fewer than 3 reads, or every rate
   identical, there is no trend to fit: persist the last rate. If
   the tracked-player count is still small (< 10,000 — launch night,
   before Exophase's first refresh), apply a **dilution prior**:
   predict the median at 0.4x the last rate for trophies below 99%
   (the 100% trophy stays ~99%), because a refresh is coming and
   it will cut the rates. A real history overrides this prior.
5. **Spread.** Random walk `0.5 * sqrt(horizon_days)` (the house
   baseline's term) combined in quadrature with the fit residual and
   a trend-uncertainty term `0.3 * |damped logit move|`, capped at
   3.0 logit units. Quantiles are the median logit plus
   `z_tau * spread` (`z` = -1.645, -0.674, 0, 0.674, 1.645),
   mapped back to percent and rounded to 4 decimals (half up).

Why this beats persisting the last rate: on the final case the
history spans ~2 days of real movement (refresh, dilution, then
players progressing), so the trend carries signal the baseline
ignores; on a flat early history the dilution prior anticipates
the refresh instead of trusting a pre-refresh rate.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with one case file in
`/input` and reads `<id>.out` from `/output`. Output format:

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

one row per trophy, each `q` five finite numbers 0–100,
non-decreasing.

## Example output (practice case p01, 2026-11-19)

The practice history is flat (638 players, pre-refresh), so the
dilution prior fires and the medians sit at 0.4x the last rate:

| Trophy | Last rate | q05 | q50 | q95 |
| --- | --- | --- | --- | --- |
| Vice Got a Price! (Platinum) | 0.63% | 0.0915 | 0.2520 | 0.6924 |
| Just Like Paradise | 100% | 99.9873 | 99.9900 | 99.9921 |
| Paradise Lost | 1.25% | 0.1818 | 0.5000 | 1.3679 |
| T**s Ass Drug Money | 1.10% | 0.1599 | 0.4400 | 1.2050 |

Full practice outputs are in [`example_output/`](example_output/).
A synthetic final-like case (rising story trophies, falling Platinum)
is used to check the trend path; see `main.py`'s docstring.

## License

MIT. See [LICENSE](LICENSE).
