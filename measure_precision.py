from train_model import estimatedPrice
import pandas as pd
from math import sqrt


def calculate_R2(sum_of_error, data: pd.DataFrame):
    """
    Calculates the Coefficient of Determination (R²) score.

    Args:
        sum_of_error (float): The Sum of Squared Errors (RSS).
        data (pd.DataFrame): The dataset used for calculating the Total Sum of Squares (SST).
    """
    mean = data['price'].mean()
    summ = 0
    for row in data.itertuples(index=False):
        summ += ((row.price - mean) ** 2)
    result_of_precision = 1 - (sum_of_error / summ)
    print(f"The precision using R2 : {result_of_precision:.2f}")
    #  It's the Coefficient of Determination


def get_precision(theta0, theta1, min_km, max_km, data: pd.DataFrame):
    """
    Calculates and prints the model precision metrics (R² and RMSE).

    Args:
        theta0 (float): Trained y-intercept.
        theta1 (float): Trained slope.
        min_km (float): Min mileage for normalization.
        max_km (float): Max mileage for normalization.
        data (pd.DataFrame): The original dataset.
    """
    index = 0
    sum_of_error = 0
    for row in data.itertuples(index=False):
        x = (row.km - min_km) / (max_km - min_km)
        sum_of_error += ((row.price - estimatedPrice(theta0, theta1, x)) ** 2)
        index += 1
    calculate_R2(sum_of_error, data)
    sum_of_error /= index
    print(f"The precision using RMSE : {sqrt(sum_of_error):.4f}")
