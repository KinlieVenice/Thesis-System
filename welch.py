import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data
data = pd.DataFrame({
    'Group': ['Manual', 'Automated'],
    'Mean': [451.00, 485.75],
    'CI_low': [437.2, 484.5],   # Your calculated lower bounds
    'CI_high': [464.8, 487.0]    # Your calculated upper bounds
})

# Create plot with error bars
plt.figure(figsize=(8, 6))
ax = sns.pointplot(
    x='Group', 
    y='Mean', 
    data=data,
    errorbar=None,  # We'll add custom error bars
    capsize=0.2,
    markers='o',
    scale=0.7
)

# Add custom error bars manually
for i, group in enumerate(data['Group']):
    y_err_lower = data.loc[i, 'Mean'] - data.loc[i, 'CI_low']
    y_err_upper = data.loc[i, 'CI_high'] - data.loc[i, 'Mean']
    plt.errorbar(
        x=i,
        y=data.loc[i, 'Mean'],
        yerr=[[y_err_lower], [y_err_upper]],  # Fixed bracket structure
        fmt='none',
        capsize=5,
        color='black',
        linewidth=2
    )

# Formatting
plt.title("Mean Comparison (Welch's t-test)\np < 0.0001", pad=20)
plt.ylabel("Measurement Value")
sns.despine()

# Add significance bar
plt.plot([0, 1], [500, 500], color='black', linewidth=1.5)
plt.text(0.5, 502, "***", ha='center', fontsize=12)

plt.tight_layout()
plt.show()