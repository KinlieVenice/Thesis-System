import seaborn as sns  
import matplotlib.pyplot as plt  
data = {'Group': ['Manual', 'Automated'], 'Mean': [451.00, 485.75], 'CI': [5.8, 1.2]}  
sns.pointplot(x='Group', y='Mean', data=data, ci='CI', capsize=0.1)  
plt.title("Mean Comparison (Welch’s t-test, p < 0.0001)")  
plt.show()  