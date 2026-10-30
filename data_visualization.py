import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleaned_car_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Graph 1: Distribution of selling prices
plt.figure(figsize=(10, 5))

plt.hist(df["selling_price"], bins=30)

plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price (₹)")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.show()


# Graph 2: Vehicle age vs selling price
plt.figure(figsize=(10, 5))

plt.scatter(
    df["vehicle_age"],
    df["selling_price"],
    alpha=0.3
)

plt.title("Vehicle Age vs Selling Price")
plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Selling Price (₹)")

plt.tight_layout()
plt.show()


# Graph 3: Kilometres driven vs selling price
plt.figure(figsize=(10, 5))

plt.scatter(
    df["km_driven"],
    df["selling_price"],
    alpha=0.3
)

plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (₹)")

plt.tight_layout()
plt.show()

print("All graphs generated successfully!")
