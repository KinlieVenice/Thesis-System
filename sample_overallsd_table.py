import numpy as np
import pandas as pd
import os

# Function to calculate mean and sample standard deviation
def calculate_sample_sd(responses):
    responses_array = np.array(responses)
    overall_mean = np.mean(responses_array)
    sample_sd = np.std(responses_array, ddof=1)  # ddof=1 for sample SD
    return overall_mean, sample_sd

# Example input: responses per question
responses_per_question = [
    [5,4,4,3,4,4,4,4,4,3,4,5,5,4],
    [4,5,5,4,4,5,5,4,4,4,4,3,4,5],
    [5,4,5,5,5,4,5,5,4,5,5,4,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,4,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,4,4,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,4,5,5,5,5,5,5,5,5,5,4,4,5],
    [5,5,5,5,5,5,5,5,5,4,5,5,5,5],
    [5,4,5,5,4,5,5,5,5,5,5,5,5,5],
    [4,5,5,4,5,5,5,5,5,5,5,4,4,5],
    [4,5,4,4,5,5,5,4,5,5,5,5,4,5],
    [5,4,5,5,4,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,4,5,4,4,5,5,4,5,4],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,4,5,5,4,5,5,4,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,4,4,5,5,5,5,5,5,4,5,5,5],
    [5,5,5,4,5,5,5,5,5,5,4,5,5,4],
    [4,5,4,4,5,5,5,5,5,5,4,5,5,5],
    [5,5,5,4,5,4,5,5,5,4,4,5,5,4],
    [5,5,5,5,5,5,5,5,4,5,5,5,5,4],
    [4,5,5,4,4,5,5,5,5,4,5,5,5,5],
    [5,5,5,4,4,5,5,5,5,4,4,5,5,5],
    [4,5,5,4,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,4,4,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,4,5,5,5,5],
    [5,5,5,4,5,5,5,5,4,5,5,5,5,4],
    [5,4,5,5,4,5,5,5,4,5,4,4,4,4],
    [5,4,5,4,5,5,5,5,4,5,4,4,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    [4,4,4,3,4,5,4,4,4,4,4,4,4,4],
    [4,4,4,3,4,4,4,4,4,4,4,4,4,4],
    [5,5,5,4,5,4,5,5,4,5,4,5,5,5],
    [5,5,4,4,4,4,5,5,4,5,4,4,4,5],
    [5,5,5,5,4,5,5,5,5,4,5,5,4,4],
    [4,4,5,4,4,4,4,4,4,4,4,4,4,4],
    [5,5,5,5,5,5,5,4,4,5,5,5,5,5],
    [4,4,5,4,4,4,4,5,4,4,4,5,4,4],
    [4,4,4,4,4,4,5,4,4,4,5,4,4,4],
    [4,5,4,4,5,4,4,4,4,4,5,4,4,4],
    [4,4,4,4,4,5,4,4,4,5,4,4,4,5],
    [4,4,4,5,4,4,4,4,4,4,4,5,4,4],
    [4,4,4,5,4,4,4,5,4,4,4,4,4,4]
]

# Store all responses together
all_responses = []

# Prepare data for DataFrame
results = []

for idx, responses in enumerate(responses_per_question):
    mean, sd = calculate_sample_sd(responses)
    results.append({
        "Question": f"Question {idx + 1}",
        "Mean": mean,
        "Sample Standard Deviation": sd
    })
    all_responses.extend(responses)

# Calculate overall mean and SD
overall_mean, overall_sd = calculate_sample_sd(all_responses)

# Add overall results as a separate row
results.append({
    "Question": "Overall",
    "Mean": overall_mean,
    "Sample Standard Deviation": overall_sd
})

# Create a DataFrame
df = pd.DataFrame(results)

# Set output path
output_folder = r"C:\Users\Huawei\OneDrive\文档\THESIS SURVEYS"
output_filename = "results.xlsx"
output_path = os.path.join(output_folder, output_filename)

# Save to Excel
df.to_excel(output_path, index=False)

print(f"Results successfully exported to '{output_path}'!")
