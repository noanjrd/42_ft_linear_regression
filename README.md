# ft_linear_regression

A linear regression implementation from scratch to predict car prices based on their mileage (kilometers). This project uses Gradient Descent for optimization and normalization for better performance.

## Project Overview 

The goal of this project is to implement a linear regression model that finds the relationship between a car's mileage and its price. The model is trained using a dataset (`data.csv`) and can then predict prices for any user-provided mileage.

### Mathematical Concepts

- **Hypothesis Function**: $h_{\theta}(x) = \theta_0 + \theta_1x$
- **Cost Function (MSE)**: $J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$
- **Gradient Descent**: Iteratively updates $\theta_0$ and $\theta_1$ to minimize the cost function.
- **Feature Scaling**: Mileage values are normalized to a range of $[0, 1]$ to ensure the gradient descent converges efficiently.

## Features

- **Linear Regression from Scratch**: No high-level ML libraries like scikit-learn used for the core logic.
- **Normalization**: Handles large mileage values automatically.
- **Visualization**: Generates a scatter plot of the data with the resulting regression line.
- **Precision Metrics**: 
    - **RMSE (Root Mean Squared Error)**: Measures the average prediction error in the original price unit.
    - **R² Score**: Indicates the goodness-of-fit (1.0 is perfect).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/noanjrd/42_ft_linear_regression
   cd 42_ft_linear_regression
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the model
Before making predictions, you need to train the model using the provided dataset. This iteratively updates the parameters using Gradient Descent:
```bash
python3 train_model.py
```
This will generate or update `thetas.json` with the optimized $\theta_0$ and $\theta_1$.

### 2. Predict a price
Run the prediction script with a mileage value (km) to get an estimate:
```bash
python3 estimate_price.py [mileage]
```

Example:
```bash
python3 estimate_price.py 150000
```
This script will:
1. Load the parameters from `thetas.json`.
2. Normalize the input mileage.
3. Display the estimated price.
4. Calculate precision metrics (RMSE & $R^2$) if the model was trained.
5. Display a plot of the data and the regression line.

- `train_model.py`: Core logic for normalization and gradient descent. Saves results to `thetas.json`.
- `estimate_price.py`: Main entry point for predictions. Loads parameters and displays a visualization.
- `measure_precision.py`: Calculates RMSE and R² metrics to evaluate the model performance.
- `data.csv`: The dataset containing mileage (`km`) and price (`price`).
- `thetas.json`: Stores the calculated $\theta_0$ and $\theta_1$.

## References & Resources

This implementation was guided by the following resources:
- [A Simple Explanation of Linear Regression](https://medium.com/@alejandro-ao/a-simple-explanation-of-linear-regression-cb6126afe4c2) by Alejandro AO.
- [Linear Regression using Gradient Descent](https://medium.com/data-science/linear-regression-using-gradient-descent-97a6c8700931) by Adarsh Menon.

**Recommendation:** To fully understand how the model updates its parameters, I strongly recommend studying how **derivatives** work, as they are the core mechanism behind Gradient Descent.

## License

This project is part of the 42 School curriculum and follows the school's academic policies.

