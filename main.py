import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_model import start_prediction


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


def main():
    """
    Main entry point of the program. 
    Parses command-line arguments to trigger price prediction based on mileage.
    """
    argv = sys.argv
    try:
        assert len(argv) == 2, "Wrong number of arguments"
        data = pd.read_csv("data.csv")
        start_prediction(data, int(argv[1]))
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
    except KeyboardInterrupt:
        print("Progam interupted")
        exit(1)
    return


if __name__ == "__main__":
    main()
