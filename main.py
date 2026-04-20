import sys
import pandas as pd


def estimatedPrice(theta_0, theta_1 ,mileage):
    return theta_0 + (theta_1 * mileage)
    
def train_on_data(data : pd.DataFrame):
    # print(data)
    theta_0 = 0
    theta_1 = 0
    for row in data.itertuples(index=False):
        theta_1 = estimatedPrice(theta_0, theta_1, row.price) * row.price
    print(theta_1)

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