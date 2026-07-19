import pandas as pd
import json


def save_thetas(theta0, theta1):
    """Saves the trained model parameters to a JSON file.

    Args:
        theta0 (float): The trained y-intercept (bias term).
        theta1 (float): The trained slope (weight term).
    """
    data = {
        "theta0": theta0,
        "theta1": theta1}
    with open("thetas.json", "w") as f:
        json.dump(data, f, indent=4)


def estimatedPrice(theta_0, theta_1, mileage):
    return theta_0 + (theta_1 * mileage)


def calculate_slop_of_MSE_theta1(theta0, theta1, min_km, max_km, data: pd.DataFrame):
    """Calculates the partial derivative of the Cost Function with respect to theta1.

    This gradient is used to update the slope (theta1) during gradient descent.
    It uses the Mean Squared Error (MSE) formula.

    Args:
        theta0 (float): Current y-intercept.
        theta1 (float): Current slope.
        min_km (float): Minimum mileage in the dataset for normalization.
        max_km (float): Maximum mileage in the dataset for normalization.
        data (pd.DataFrame): The training dataset containing 'km' and 'price'.

    Returns:
        float: The mean of the errors multiplied by normalized mileage.
    """

    x = (data["km"] - min_km) / (max_km - min_km)
    errors = x * (estimatedPrice(theta0, theta1, x) - data["price"])
    return errors.mean()
    # In the real derivative, we write -2 and not 1


def calculate_slop_of_MSE_theta0(theta0, theta1, min_km, max_km, data: pd.DataFrame):
    """Calculates the partial derivative of the Cost Function with respect to theta0.

    This gradient is used to update the y-intercept (theta0) during gradient descent.
    It uses the Mean Squared Error (MSE) formula.

    Args:
        theta0 (float): Current y-intercept.
        theta1 (float): Current slope.
        min_km (float): Minimum mileage in the dataset for normalization.
        max_km (float): Maximum mileage in the dataset for normalization.
        data (pd.DataFrame): The training dataset containing 'km' and 'price'.

    Returns:
        float: The mean of the differences between predictions and actual prices.
    """
    x = (data["km"] - min_km)/(max_km - min_km)
    errors = estimatedPrice(theta0, theta1, x) - data["price"]
    return errors.mean()


def start_training(data: pd.DataFrame):
    """Trains the linear regression model using Gradient Descent.

    The function normalizes the features (km) to improve convergence speed and
    iteratively updates theta0 and theta1 to minimize the cost function.

    Args:
        data (pd.DataFrame): The dataset containing 'km' (mileage) and 'price'.
    """
    theta_0 = 0
    theta_1 = 0
    min_km = data['km'].min()
    max_km = data['km'].max()
    learning_rate = 0.1
    epoch = 1000
    for _ in range(epoch):
        res_derivative_MSE_theta1 = calculate_slop_of_MSE_theta1(theta_0, theta_1, min_km, max_km, data)
        res_derivative_MSE_theta0 = calculate_slop_of_MSE_theta0(theta_0, theta_1, min_km, max_km, data)
        theta_0 = theta_0 - (learning_rate * res_derivative_MSE_theta0)
        theta_1 = theta_1 - (learning_rate * res_derivative_MSE_theta1)
    save_thetas(theta_0, theta_1)


def main():
    """Main execution block for training the model.

    Reads the dataset from data.csv, initiates the training process,
    and handles potential errors during execution.
    """
    try:
        data = pd.read_csv("data.csv")
        start_training(data)
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
