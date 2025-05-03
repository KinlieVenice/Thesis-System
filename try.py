import numpy as np
from scipy import stats

# Your list of responses (already provided)
responses = [4.75, 4.75, 4.8, 4.55, 4.7, 4.766666667, 4.8, 4.8, 4.666666667, 4.7, 4.7, 4.733333333,	4.733333333, 4.733333333]

# Convert the list of lists to a NumPy array
responses_array = np.array(responses)

# Flatten to a 1D array for overall Likert score analysis
overall_scores = responses_array.flatten()

# Hypothesized mean (e.g., neutral = 3 on a 5-point scale)
mu = 3

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(overall_scores, mu)

# Right-tailed adjustment
if t_statistic > 0:
    p_value_right = p_value / 2
else:
    p_value_right = 1 - (p_value / 2)

# Decision
alpha = 0.05
decision = "Reject null hypothesis" if p_value_right < alpha else "Fail to reject null hypothesis"

# Output
print(f"T-statistic: {t_statistic}")
print(f"Right-tailed p-value: {p_value_right}")
print(f"Decision: {decision}")
