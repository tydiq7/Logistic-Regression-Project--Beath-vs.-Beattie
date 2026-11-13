# Logistic Regression Project: Beath vs. Beattie
Motivation: Roommate and I have a difference in opinion over who is the better (most beneficial for the team) player on the UCSD basketball team, Leo Beath, or Tom Beattie.

Goal: Figure out which player is more impactful for the team based on their stats per game.

Belief: Assuming one player has a greater impact on the team than the other, their performance should be a better predictor of whether or not the team wins or loses, than the other player. So, considering each player's actual stats in each game relative to their mean stats (in order to determine whether they over or underperformed), and then correlating that with whether or not the team won or lost, should reveal information about which player’s performance is more important to the team’s success.

Methodology: Because I am trying to assess which player’s performance is a better indicator of the team’s success, I am essentially comparing two independent variables to a predicted binary outcome, winning or losing. Therefore, I believe the best fitting machine learning model would be a logistic regression model. If the number of datapoints for each player is unsatisfactory to draw conclusions, bootstrapping could be utilized to calculate the accuracy of the model with limited data.

Skills: Hopefully, at the end of this project, I have improved my skills in cleaning and processing datasets, as well as developed an understanding of how logistic regression machine learning models work, and when they can be implemented.

## Running

```
pip install numpy pandas matplotlib seaborn scikit-learn
python main.py
```

Paths are relative to `main.py`, so it works from any clone. Figures are written to `Figures/`.

## Results

Evaluation: game-grouped 5-fold cross-validation (both halves of a game always sit in the same fold), 45 halves from 23 games. Every metric is computed on held-out predictions only. Values are mean ± std across folds.

| Metric | Model | Baseline (majority class) |
|---|---|---|
| AUC | 0.711 ± 0.213 | 0.500 ± 0.000 |
| Accuracy | 0.646 ± 0.103 | 0.618 ± 0.076 |
| Log-loss (lower is better) | 0.647 ± 0.165 | 0.670 ± 0.042 |

Standardized coefficients (full-data fit, for interpretation only): L_Assists 0.735, T_Points 0.721, T_Assists 0.655, T_Rebounds 0.569, L_Points 0.301, L_Rebounds -0.198.

Takeaway: the model beats the baseline on all three metrics, but only modestly, and the fold-to-fold spread is large. With 23 games, the data can't separate the two players with any confidence. The coefficients are suggestive, not conclusive.

### Why the earlier 1.0 AUC was misleading

The first version used a single 80/20 split and reported AUC = 1.0 and accuracy = 1.0. That number should not be trusted, for three reasons:

1. **Tiny test set.** 20% of 45 rows is 9 halves (7 wins, 2 losses). With 2 losses, a perfect ranking is easy to get by luck.
2. **Game leakage.** Each game contributes two rows (one per half), and the split was random by row. One half of a game could be in training while the other half was in the test set, so the model had already seen that game.
3. **Optimistic log-loss.** Log-loss was computed on the full dataset, training rows included, so it looked better than the model really is.

The current evaluation splits by game, scores only held-out predictions, and fits the standardization inside each fold. It also reports a baseline, so the numbers can be read in context.

## Next Steps: Extend this concept to finance and investing. For example, comparing two technical indicators to determine which is a better predictor of a certain stock or index price.
