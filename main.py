import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_info(data, theta0, theta1, min_km, max_km):
    ax = data.plot(kind='scatter', x='km', y='price')
    x_real = np.linspace(min_km, max_km, 100)

    x_norm = (x_real - min_km) / (max_km - min_km)
    y = theta0 + theta1 * x_norm
    
    ax.plot(x_real, y, color='red')
    plt.show()


def estimatedPrice(theta_0, theta_1 ,mileage):
    return theta_0 + (theta_1 * mileage)


def calculate_derivative_of_MSE_theta1(theta0, theta1,min_km, max_km, data : pd.DataFrame):
    index = 0
    summ = 0
    for row in data.itertuples(index=False):
        x = (row.km - min_km) / (max_km - min_km)
        summ += x * (estimatedPrice(theta0, theta1, x) - row.price )
        index += 1
    return (1 / index) * summ
    # In the real derivative, we write -2 and not 1


def calculate_derivative_of_MSE_theta0(theta0, theta1,min_km, max_km, data : pd.DataFrame):
    index = 0
    summ = 0
    for row in data.itertuples(index=False):
        x = (row.km - min_km) / (max_km - min_km)
        summ += estimatedPrice(theta0, theta1, x) - row.price 
        index+=1
    return (1 / index) * summ


def train_on_data(data : pd.DataFrame, input_value):
    theta_0 = 0
    theta_1 = 0
    min_km = data['km'].min()
    max_km = data['km'].max()
    learning_rate = 0.1
    for _ in range(10000):
        MSE_theta1 = calculate_derivative_of_MSE_theta1(theta_0, theta_1, min_km, max_km,data)
        MSE_theta0 = calculate_derivative_of_MSE_theta0(theta_0, theta_1, min_km, max_km, data)
        theta_0 = theta_0 - (learning_rate * MSE_theta0)
        theta_1 = theta_1 - (learning_rate * MSE_theta1)

    x = (input_value - min_km) / (max_km - min_km)
    estimated_price_result = estimatedPrice(theta_0, theta_1, x)
    print("Here is the estimated price :", estimated_price_result)
    display_info(data,theta_0, theta_1 , min_km, max_km)
    


def main():
    argv = sys.argv
    try:
        assert len(argv) == 2, "Wrong number of arguments"
        data =  pd.read_csv("data.csv")
        train_on_data(data, int(argv[1]))
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
    except KeyboardInterrupt as e:
        print("Progam interupted")
        exit(1)
        
    return

if __name__ == "__main__":
    main()