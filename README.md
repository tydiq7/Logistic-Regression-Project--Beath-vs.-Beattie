# Logistic Regression Project: Beath vs. Beattie
Motivation: Roommate and I have a difference in opinion over who is the better (most beneficial for the team) player on the UCSD basketball team, Leo Beath, or Tom Beattie.

Goal: Figure out which player is more impactful for the team based on their stats per game.

Belief: Assuming one player has a greater impact on the team than the other, their performance should be a better predictor of whether or not the team wins or loses, than the other player. So, considering each player's actual stats in each game relative to their mean stats (in order to determine whether they over or underperformed), and then correlating that with whether or not the team won or lost, should reveal information about which player’s performance is more important to the team’s success.

Methodology: Because I am trying to assess which player’s performance is a better indicator of the team’s success, I am essentially comparing two independent variables to a predicted binary outcome, winning or losing. Therefore, I believe the best fitting machine learning model would be a logistic regression model. If the number of datapoints for each player is unsatisfactory to draw conclusions, bootstrapping could be utilized to calculate the accuracy of the model with limited data.

Skills: Hopefully, at the end of this project, I have improved my skills in cleaning and processing datasets, as well as developed an understanding of how logistic regression machine learning models work, and when they can be implemented.

Next Steps: Extend this concept to finance and investing. For example, comparing two technical indicators to determine which is a better predictor of a certain stock or index price.
