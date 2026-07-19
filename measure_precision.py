from train_model import estimatedPrice
import pandas as pd
from math import sqrt


def calculate_R2(sum_of_errors, data: pd.DataFrame):
    """
    Calculates the Coefficient of Determination (R²) score.

    Args:
        sum_of_error (float): The Sum of Squared Errors (RSS).
        data (pd.DataFrame): The dataset used for calculating the Total Sum of Squares (SST).

    Returns:
        None: The function prints the R2 score.
    """
    mean = data['price'].mean()
    summ = ((data['price'] - mean) ** 2).sum()
    result_of_precision = 1 - (sum_of_errors / summ)
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
    data_normalized = data.copy()
    data_normalized['km'] = (data_normalized['km'] - min_km) / (max_km - min_km)
    squared_error = ((data_normalized['price'] - estimatedPrice(theta0, theta1, data_normalized['km'])) ** 2)
    calculate_R2(squared_error.sum(), data)
    print(f"The precision using RMSE : {sqrt(squared_error.mean()):.4f}")
