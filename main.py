import pandas as pd

# Load the dataset
df = pd.read_csv("car_data.csv")

# Display the first five rows
print(df.head())

# Display dataset size
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset Information:")
df.info()