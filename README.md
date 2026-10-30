# Car Price Prediction Using Python

## Project Overview

This project predicts the estimated selling price of a used car using machine learning. It uses a Random Forest Regression model trained on historical car data.

The project includes data cleaning, exploratory data visualization, model training, evaluation, and a Streamlit web application.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## Project Features

- Clean and analyze used-car data.
- Visualize relationships between car features and selling prices.
- Train a Random Forest Regression model.
- Evaluate predictions using MAE, RMSE, and R².
- Predict a car's estimated selling price through a web interface.

## Project Structure

```text
Car_Price_Prediction/
├── main.py
├── data_cleaning.py
├── data_visualization.py
├── model_training.py
├── predict_car_price.py
├── app.py
├── car_price_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

1. Install Python and Git.
2. Clone or download this repository.
3. Create and activate a Python virtual environment.
4. Install the required packages:

   `pip install -r requirements.txt`

5. Obtain the dataset and save it as `car_data.csv`.
6. Run the data-cleaning script:

   `python data_cleaning.py`

7. Generate the visualizations:

   `python data_visualization.py`

8. Train the model if needed:

   `python model_training.py`

9. Start the web application:

   `python -m streamlit run app.py`

## Model Performance

Initial evaluation results:

- MAE: ₹103,361.91
- RMSE: ₹262,616.43
- R²: 0.8866

These results are based on the initial test split and may change when the model or dataset changes.

## Limitations

Predictions are estimates based on historical data and are not guaranteed market valuations. Prediction quality depends on the dataset and the characteristics of the car being evaluated.