import pandas as pd

# Load the dataset
df = pd.read_csv("car_data.csv")

print("Original dataset shape:", df.shape)

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing selling prices
df = df.dropna(subset=["selling_price"])

# Remove records with invalid selling prices
df = df[df["selling_price"] > 0]

# Display cleaned dataset information
print("\nCleaned dataset shape:", df.shape)

print("\nCleaned dataset information:")
df.info()

print("\nFirst five cleaned records:")
print(df.head())

# Save the cleaned dataset
df.to_csv("cleaned_car_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
