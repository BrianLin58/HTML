import pandas as pd

# Load the dataset
data = pd.read_csv("preprocess_test_1.csv")

# Convert columns 11 onward to numeric (coerce invalid values to NaN)
for column in data.columns[11:]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Calculate column averages (ignoring NaN)
column_averages = data.iloc[:, 11:].mean()

# Fill missing values (NaN) with column averages
data.iloc[:, 11:] = data.iloc[:, 11:].fillna(column_averages)

# Save the updated DataFrame to a new CSV file
data.to_csv("preprocess_test_2.csv", index=False)
