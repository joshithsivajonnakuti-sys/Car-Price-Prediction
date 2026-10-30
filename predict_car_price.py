import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load("car_price_model.pkl")

# Enter the details of the car
car = pd.DataFrame([
    {
        "brand": "Maruti",
        "vehicle_age": 5,
        "km_driven": 40000,
        "seller_type": "Individual",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "mileage": 20.0,
        "engine": 1197,
        "max_power": 82.0,
        "seats": 5
    }
])

# Predict the selling price
predicted_price = model.predict(car)[0]

# Display the result
print("\n===== CAR PRICE PREDICTION =====")

print("Car Brand:", car.loc[0, "brand"])
print("Vehicle Age:", car.loc[0, "vehicle_age"], "years")
print("Kilometres Driven:", car.loc[0, "km_driven"], "km")

print(f"\nEstimated Selling Price: ₹{predicted_price:,.0f}")

print("================================")
