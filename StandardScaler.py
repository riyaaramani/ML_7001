# Practical: Applying StandardScaler in Python
# StandardScaler standardizes numerical data to mean 0 and standard deviation 1.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Create the StandardScaler object
scaler = StandardScaler()

# Standardize the numerical columns
scaled_array = scaler.fit_transform(df)

# Convert the scaled values back to a DataFrame
scaled_df = pd.DataFrame(scaled_array, columns=df.columns)

print("\nData after applying StandardScaler:")
print(scaled_df.round(2))

# Check the mean and standard deviation after scaling
print("\nMeans after scaling:")
print(scaled_df.mean().round(2))

print("\nStandard deviations after scaling:")
print(scaled_df.std(ddof=0).round(2))
