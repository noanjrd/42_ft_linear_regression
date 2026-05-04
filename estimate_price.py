import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import sys


def display_info(data, theta0, theta1, min_km, max_km):
    """
    Displays the dataset as a scatter plot and overlays the regression line.

    Args:
        data (pd.DataFrame): The dataset containing 'km' and 'price' columns.
        theta0 (float): The y-intercept of the regression model.
        theta1 (float): The slope of the regression model.
        min_km (float): The minimum mileage for normalization.
        max_km (float): The maximum mileage for normalization.
    """
    ax = data.plot(kind='scatter', x='km', y='price')
    x_real = np.linspace(min_km, max_km, 100)
    x_norm = (x_real - min_km) / (max_km - min_km)
    y = theta0 + theta1 * x_norm
    ax.plot(x_real, y, color='red', label="Linear Regression line")
    plt.title("Price of cars according to their mileage")
    plt.xlabel("Mileage (in km)")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def estimatedPrice(theta_0, theta_1, mileage):
    """
    Calculates the estimated price for a given normalized mileage.

    Args:
        theta_0 (float): The y-intercept (bias).
        theta_1 (float): The slope (weight).
        mileage (float): The normalized mileage value.

    Returns:
        float: The predicted price.
    """
    return theta_0 + (theta_1 * mileage)


def get_thetas():
    """
    Loads trained model parameters from the JSON file.

    Returns:
        tuple[float, float]: A tuple containing (theta0, theta1). If loading fails,
        returns (0, 0).
    """
    try:
        with open("thetas.json", "r") as f:
            data = json.load(f)
        theta0 = data['theta0']
        theta1 = data['theta1']
        thetas_in_file = True
    except:
        theta0 = 0
        theta1 = 0
        thetas_in_file = False
    return theta0, theta1, thetas_in_file


def predict_value(data, input_value):
    """
    Normalizes a mileage input and prints the predicted car price.

    If trained parameters are available, it also prints evaluation metrics and
    displays the regression chart.

    Args:
        data (pd.DataFrame): Dataset containing 'km' and 'price' columns.
        input_value (int | float): Raw mileage value provided by the user.
    """
    min_km = data['km'].min()
    max_km = data['km'].max()
    x = (input_value - min_km) / (max_km - min_km)
    theta0, theta1, thetas_in_file = get_thetas()
    estimated_price_result = estimatedPrice(theta0, theta1, x)
    if estimated_price_result < 0:
        estimated_price_result = 0
    print(f"The estimated price : {estimated_price_result:.4f}")
    from measure_precision import get_precision
    if (thetas_in_file):
        get_precision(theta0, theta1, min_km, max_km, data)
        display_info(data, theta0, theta1, min_km, max_km)


def main():
    """
    Main entry point of the program.
    Parses command-line arguments to trigger price prediction based on mileage.
    """
    argv = sys.argv
    try:
        assert len(argv) == 2, "Wrong number of arguments"
        assert int(argv[1]) >= 0, "Mileage value must be positive"
        data = pd.read_csv("data.csv")
        predict_value(data, int(argv[1]))
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print("Error :", e)
        exit(1)
    except KeyboardInterrupt:
        print("Progam interupted")
        exit(1)
    return


if __name__ == "__main__":
    main()
