import pandas as pd
from scipy import stats

# Load the survey data (replace with your actual file or DataFrame)
df = pd.read_csv(r"C:\Users\Huawei\OneDrive\文档\THESIS SURVEYS\software_result.csv")  # Replace with your CSV file path

# Set the hypothesized mean for the Likert scale (neutral value = 3)
mu_0 = 3

# Flatten the data by converting the survey responses into a single list
combined_responses = df.values.flatten()  # Flatten the DataFrame to a 1D array

# Perform the one-sample t-test on the combined responses
t_stat, p_val_two_tailed = stats.ttest_1samp(combined_responses, mu_0)

# Adjust p-value for one-tailed test
if t_stat > 0:
    p_val_one_tailed = p_val_two_tailed / 2  # If t-statistic is positive
else:
    p_val_one_tailed = 1 - (p_val_two_tailed / 2)  # Rare case, if t-statistic is negative

# Print the results
print("Overall t-statistic:", t_stat)
print("Overall One-tailed p-value:", p_val_one_tailed)
