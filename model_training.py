import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load the cleaned dataset
df = pd.read_csv("cleaned_car_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Select input features
features = [
    "brand",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

target = "selling_price"

X = df[features]
y = df[target]

# Step 3: Identify numerical and categorical columns
numeric_features = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

categorical_features = [
    "brand",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

# Step 4: Handle missing values and encode categories
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Step 5: Create the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 7: Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Step 8: Make predictions on test data
predictions = model.predict(X_test)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print("------------------------")
print(f"Mean Absolute Error: ₹{mae:,.2f}")
print(f"Root Mean Squared Error: ₹{rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Step 10: Save the trained model
joblib.dump(model, "car_price_model.pkl")

print("\nTrained model saved successfully!")
print("File created: car_price_model.pkl")
