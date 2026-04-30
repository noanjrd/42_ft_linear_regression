import pandas as pd


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


def calculate_slop_of_MSE_theta1(theta0, theta1, min_km, max_km, data: pd.DataFrame):
    """
    Calculates the partial derivative of the Mean Squared Error with respect to theta1.

    Args:
        theta0 (float): Current y-intercept.
        theta1 (float): Current slope.
        min_km (float): Min mileage for normalization.
        max_km (float): Max mileage for normalization.
        data (pd.DataFrame): The training dataset.

    Returns:
        float: The gradient for theta1.
    """
    index = 0
    summ = 0
    for row in data.itertuples(index=False):
        x = (row.km - min_km) / (max_km - min_km)
        summ += x * (estimatedPrice(theta0, theta1, x) - row.price)
        index += 1
    return (1 / index) * summ
    # In the real derivative, we write -2 and not 1


def calculate_slop_of_MSE_theta0(theta0, theta1, min_km, max_km, data: pd.DataFrame):
    """
    Calculates the partial derivative of the Mean Squared Error with respect to theta0.

    Args:
        theta0 (float): Current y-intercept.
        theta1 (float): Current slope.
        min_km (float): Min mileage for normalization.
        max_km (float): Max mileage for normalization.
        data (pd.DataFrame): The training dataset.

    Returns:
        float: The gradient for theta0.
    """
    index = 0
    summ = 0
    for row in data.itertuples(index=False):
        x = (row.km - min_km) / (max_km - min_km)
        summ += estimatedPrice(theta0, theta1, x) - row.price
        index += 1
    return (1 / index) * summ


def start_prediction(data: pd.DataFrame, input_value):
    """
    Trains the linear regression model using gradient descent and predicts the price for a given mileage.

    Args:
        data (pd.DataFrame): The dataset containing 'km' and 'price'.
        input_value (int): The mileage value to predict the price for.
    """
    theta_0 = 0
    theta_1 = 0
    min_km = data['km'].min()
    max_km = data['km'].max()
    learning_rate = 0.1
    round = 1000
    for _ in range(round):
        res_derivative_MSE_theta1 = calculate_slop_of_MSE_theta1(theta_0, theta_1, min_km, max_km, data)
        res_derivative_MSE_theta0 = calculate_slop_of_MSE_theta0(theta_0, theta_1, min_km, max_km, data)
        theta_0 = theta_0 - (learning_rate * res_derivative_MSE_theta0)
        theta_1 = theta_1 - (learning_rate * res_derivative_MSE_theta1)

    x = (input_value - min_km) / (max_km - min_km)
    estimated_price_result = estimatedPrice(theta_0, theta_1, x)
    print(f"The estimated price : {estimated_price_result:.4f}")
    from measure_precision import get_precision
    get_precision(theta_0, theta_1, min_km, max_km, data)
    from main import display_info
    display_info(data, theta_0, theta_1, min_km, max_km)
