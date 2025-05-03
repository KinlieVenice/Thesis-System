import numpy as np

# Function to calculate the sample standard deviation for responses
def calculate_sample_sd(responses):
    responses_array = np.array(responses)
    overall_mean = np.mean(responses_array)
    squared_differences = (responses_array - overall_mean) ** 2
    variance = np.sum(squared_differences) / (len(responses_array) - 1)
    sample_sd = np.sqrt(variance)
    return overall_mean, sample_sd

# Example input: array of responses per question
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

# Store all responses together for overall computation
all_responses = []

# Loop through the responses of each question
for idx, responses in enumerate(responses_per_question):
    mean, sd = calculate_sample_sd(responses)
    print(f"Question {idx+1}: Mean = {mean}, Sample Standard Deviation = {sd}")
    all_responses.extend(responses)  # Add responses to all_responses list

# Now calculate overall mean and sample SD for all responses combined
overall_mean, overall_sd = calculate_sample_sd(all_responses)

print("\n--- Overall ---")
print(f"Overall Mean = {overall_mean}")
print(f"Overall Sample Standard Deviation = {overall_sd}")
