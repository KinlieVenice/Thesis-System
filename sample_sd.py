import numpy as np

# Function to calculate the sample standard deviation for responses
def calculate_sample_sd(responses):
    # Convert responses to numpy array for easy calculations
    responses_array = np.array(responses)
    
    # Calculate the overall mean
    overall_mean = np.mean(responses_array)
    
    # Calculate the squared differences from the mean
    squared_differences = (responses_array - overall_mean) ** 2
    
    # Calculate the sample variance (degrees of freedom = n-1)
    variance = np.sum(squared_differences) / (len(responses_array) - 1)
    
    # Calculate the sample standard deviation
    sample_sd = np.sqrt(variance)
    
    return overall_mean, sample_sd

# Example input: array of responses per question
responses_per_question = [
    [4, 3, 5, 4, 3],  # Question 1
    [3, 3, 4, 2, 5],  # Question 2
    [4, 4, 3, 5, 4]   # Question 3
]

# Loop through the responses of each question and calculate the sample standard deviation
for idx, responses in enumerate(responses_per_question):
    mean, sd = calculate_sample_sd(responses)
    print(f"Question {idx+1}: Mean = {mean}, Sample Standard Deviation = {sd}")
    # print(f"Question {idx+1}: Mean = {mean:.2f}, Sample Standard Deviation = {sd:.2f}")
