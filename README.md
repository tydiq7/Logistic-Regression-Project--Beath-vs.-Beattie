# Logistic Regression Project: Beath vs. Beattie

## Answer

**Tom Beattie's over/under-performance in a half is a much better predictor of UCSD winning that half than Leo Beath's, and this is not explained by opponent strength.**

| Features (game-grouped 5-fold CV, mean AUC over 30 fold shuffles) | All 23 games | D1 opponents only (21 games) |
|---|---|---|
| Leo only | 0.60 | 0.54 |
| Tom only | 0.74 | 0.79 |
| Both players | 0.75 | 0.76 |
| Opponent strength only | 0.66 | 0.63 |
| Tom + opponent strength | 0.81 | 0.78 |
| Both players + opponent strength | 0.83 | 0.78 |

- Adding Leo to Tom barely changes the AUC (0.74 → 0.75). Adding Tom to Leo is a big jump (0.60 → 0.75). Tom's stats carry almost all of the signal.
- Opponent strength predicts wins on its own (weaker opponents → more UCSD wins), but it is almost uncorrelated with Tom's scoring (r = 0.04). So Tom's numbers are not just a stand-in for easy opponents.

**What this does and doesn't show.** This measures *predictive power*, not *importance*. A player whose output barely varies can be essential and still predict little, because the model only learns from variation. And with 23 games, every number here has wide error bars. The defensible claim is the narrow one in bold above.

## Motivation

My roommate and I disagree on who is more beneficial to the UCSD basketball team: Leo Beath or Tom Beattie.

**Idea:** if one player matters more, his over- or under-performance relative to his own average should be a better predictor of whether the team wins. Each row in the data is one half of a game, with each player's points, rebounds and assists in that half, and whether UCSD won that half.

## Methodology

- **Model:** logistic regression (binary outcome: UCSD won or lost the half). Each stat is standardized, i.e. measured in standard deviations above or below that stat's average, so stats on different scales are comparable.
- **Evaluation:** game-grouped 5-fold cross-validation. Both halves of a game always sit in the same fold, because halves from the same game share context (same opponent, same venue) and shouldn't be split across training and test data. Every metric is computed on held-out predictions only. The standardization is fit inside each fold, so the held-out games don't influence it.
- **Baseline:** a model that always predicts UCSD's win rate. Any real model has to beat this.
- **Ablation:** fit separate models on different feature groups (Leo only, Tom only, opponent strength, combinations) and compare held-out AUC. Comparing whole feature groups this way is more reliable than reading the six coefficients of a single model, which are unstable with 45 rows.

### Opponent strength (control variable)

A weak opponent makes a UCSD win more likely *and* makes it easier for any player to have a big half, so opponent strength could confound the comparison. To control for it, `Opp_Strength` is each opponent's 2025-26 win %, **with all of its games against UCSD removed** so the outcome being predicted isn't baked into the feature.

- Full-season records are used instead of the previous season because college rosters turn over heavily, so last year's record would measure this year's opponent poorly.
- La Verne (D3) and Stanton (NAIA) have no comparable Division I record. They are assigned a strength of 0.20 (below the weakest D1 opponent, 0.267), and the ablation is re-run without those two games as a robustness check. The conclusion holds either way.
- Win % is a crude measure because it ignores who each opponent played. A schedule-adjusted rating (e.g. KenPom) would be a better control.

## Results: original model (both players)

45 halves from 23 games (one tied half excluded). Mean ± std across folds:

| Metric | Model | Baseline (majority class) |
|---|---|---|
| AUC | 0.711 ± 0.213 | 0.500 ± 0.000 |
| Accuracy | 0.646 ± 0.103 | 0.618 ± 0.076 |
| Log-loss (lower is better) | 0.647 ± 0.165 | 0.670 ± 0.042 |

The model beats the baseline on all three metrics, but modestly, and the fold-to-fold spread is large.

### Why the earlier 1.0 AUC was misleading

The first version used a single 80/20 split and reported AUC = 1.0 and accuracy = 1.0. That shouldn't be trusted:

1. **Tiny test set.** 20% of 45 rows is 9 halves (7 wins, 2 losses). With only 2 losses, a perfect ranking is easy to get by luck. Across 200 random splits, single-split AUC ranged from 0.25 to 1.0.
2. **Halves from the same game split across train and test.** The split was random by row, so one half of a game could be in training while the other was in the test set. The outcome is per half (the two halves had different results in 11 of 23 games), so this is a mild leak rather than a severe one, but the halves still share context like the opponent.
3. **Optimistic log-loss.** Log-loss was computed on the full dataset, training rows included.

## Data

`data/ucsd_halves_2025_26.csv`: one row per half for UCSD's first 23 games of 2025-26 (Nov 3 - Jan 31).

| Columns | Meaning |
|---|---|
| `GameID`, `Half`, `Outcome` | ESPN game ID, half (1 or 2), UCSD result in that half (W / L / T) |
| `L_*`, `T_*` | Leo's and Tom's points, rebounds, assists in that half |
| `Date`, `Opponent`, `Site`, `Division` | Game details |
| `Opp_W`, `Opp_L` | Opponent's full 2025-26 record (incl. postseason) |
| `Opp_W_vs_UCSD`, `Opp_L_vs_UCSD` | Opponent's record against UCSD that season (removed from the strength measure) |
| `Opp_WinPct_exUCSD` | Opponent win % excluding games vs. UCSD (blank for non-D1) |
| `Opponent_Source` | How the GameID-to-opponent match was made: confirmed from ESPN's game page, inferred from the ESPN game ID sequence, or by elimination |

Sources: ESPN box scores and game pages, team athletics sites, and season pages for team records.

## Running

```
pip install numpy pandas matplotlib seaborn scikit-learn
python main.py
```

Paths are relative to `main.py`, so it works from any clone. Figures are written to `Figures/`.

## Next steps

- Replace win % with a schedule-adjusted opponent rating.
- Use play-by-play data to compute each player's on/off plus-minus, which measures importance more directly than box-score stats.
- Extend the idea to finance: e.g. compare two technical indicators as predictors of a stock or index's direction.
