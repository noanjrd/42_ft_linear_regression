# ft_linear_regression

A simple linear regression implementation from scratch to predict car prices based on their mileage (kilometers). This project uses Gradient Descent for optimization and normalization for better performance.

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

1. Clone the repository.
2. Ensure you have the required Python packages installed:
   ```bash
   pip install pandas numpy matplotlib
   ```

## Usage

To run the program and predict a price:

```bash
python main.py [mileage]
```

Example:
```bash
python main.py 150000
```

## Project Structure

- `main.py`: Entry point. Handles arguments, data loading, and starts the process.
- `train_model.py`: Core logic for normalization, gradient descent, and training.
- `measure_precision.py`: Calculates RMSE and R² metrics to evaluate the model.
- `data.csv`: The dataset containing mileage (`km`) and price (`price`).

## References & Resources

This implementation was guided by the following resources:
- [A Simple Explanation of Linear Regression](https://medium.com/@alejandro-ao/a-simple-explanation-of-linear-regression-cb6126afe4c2) by Alejandro AO.
- [Linear Regression using Gradient Descent](https://medium.com/data-science/linear-regression-using-gradient-descent-97a6c8700931) by Adarsh Menon.

**Recommendation:** To fully understand how the model updates its parameters, I strongly recommend studying how **derivatives** work, as they are the core mechanism behind Gradient Descent.

## License

This project is part of the 42 School curriculum and follows the school's academic policies.

