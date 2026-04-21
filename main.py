import sys
import pandas as pd


def estimatedPrice(theta_0, theta_1 ,mileage):
    return theta_0 + (theta_1 * mileage)


def calculate_sum__of_error(theta0, theta1, i, data : pd.DataFrame):
    index = 1
    summ = 0
    for row in data.itertuples(index=False):
        if i <= index:
            return summ
        summ += (estimatedPrice(theta0, theta1, row.km) - row.price)
        index+=1
    return summ


def train_on_data(data : pd.DataFrame):
    # print(data)
    theta_0 = 0
    theta_1 = 0
    learning_rate = 0.01
    i = 1
    for _ in range(100):
        i = 1
        for row in data.itertuples(index=False):
            sum_of_error = calculate_sum__of_error(theta_0, theta_1, i, data)
            final_error = sum_of_error * (1 / i)
            theta_0 = theta_0 - (learning_rate * final_error)
            theta_1 = theta_1 - (learning_rate * final_error)
            i+=1
            print("here",theta_1)
    print(estimatedPrice(theta_0, theta_1, 500))

def main():
    argv = sys.argv
    print(argv)
    try:
        assert len(argv) == 2, "Wrong number of arguments"
        data =  pd.read_csv("data.csv")
        train_on_data(data)
    except AssertionError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
        
    return

if __name__ == "__main__":
    main()