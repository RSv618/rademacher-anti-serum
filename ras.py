import numpy as np
import pandas as pd
from datetime import datetime


def rademacher_complexity(returns_matrix: np.ndarray, n_random_vectors: int = 1000) -> float:
    """
    Calculates the empirical Rademacher Complexity of a matrix representing returns of multiple strategies.

    Args:
        returns_matrix (np.ndarray): A matrix (T x N) representing returns where rows are time observations and columns are different strategies.
        n_random_vectors (int): The number of random Rademacher vectors to generate for complexity estimation.

    Returns:
        float: The estimated empirical Rademacher Complexity, reflecting the degree of performance sensitivity to random noise.
    """
    n_time_periods: int = len(returns_matrix)
    choices: np.ndarray = np.array([-1.0, 1.0], dtype=np.float64)

    # Generate multiple Rademacher vectors
    rademacher_vectors: np.ndarray = np.random.choice(choices, size=(n_random_vectors, n_time_periods))

    # Transpose the matrix to iterate over each column
    returns_matrix = np.transpose(returns_matrix)

    # Rademacher complexity calculation
    complexity: float = float(np.mean(
        [np.max([np.abs(np.dot(epsilon, returns_col)) / n_time_periods for returns_col in returns_matrix])
         for epsilon in rademacher_vectors]))
    return complexity


def ras_sharpe_adjustment(sharpe_ratios: pd.Series, complexity: float,
                          n_time_periods: float, n_strategies: float, delta: float=0.1) -> pd.Series:
    """
    Calculates RAS-adjusted Sharpe Ratios for a set of strategies to account for overfitting bias.

    Args:
        sharpe_ratios (pd.Series): A series of empirical Sharpe Ratios for each strategy.
        complexity (float): The empirical Rademacher Complexity.
        n_time_periods (float): The number of time periods (observations).
        n_strategies (float): The number of strategies.
        delta (float): Confidence level (default: 0.1) for the estimation error adjustment.

    Returns:
        pd.Series: The RAS-adjusted Sharpe Ratios.
    """
    estimation_error: float = (3 * np.sqrt((2 * np.log(2 / delta)) / n_time_periods) +
                               np.sqrt((2 * np.log(2 * n_strategies / delta)) / n_time_periods))
    adjusted_sharpe_ratios: pd.Series = sharpe_ratios - 2 * complexity - estimation_error
    return adjusted_sharpe_ratios


def main():
    # Load your DataFrame of log returns
    returns_df: pd.DataFrame = pd.read_csv('log_returns_matrix.csv', index_col='timestamp')

    # Calculate Sharpe Ratios for each strategy
    risk_free_rate: float | np.ndarray = 0.0
    excess_returns_df: pd.DataFrame = returns_df - risk_free_rate
    sharpe_ratios_array: pd.Series  = excess_returns_df.mean() / excess_returns_df.std()

    # Annualized sharpe ratio (optional step)
    start_date: datetime = pd.to_datetime(returns_df.index[0])
    end_date: datetime = pd.to_datetime(returns_df.index[-1])
    sqrt_n: float = np.sqrt(returns_df.shape[0] / ((end_date - start_date).total_seconds()/(365.24*24*60*60)))
    print(f"Empirical Sharpe Ratios: {sharpe_ratios_array * sqrt_n}")

    # Convert to NumPy array
    returns_matrix: np.ndarray = returns_df.values

    # Calculate empirical Rademacher Complexity
    complexity: float = rademacher_complexity(returns_matrix)

    # Apply RAS adjustment for Sharpe Ratios
    n_time_periods, n_strategies = returns_matrix.shape
    adjusted_sharpe_ratios: pd.Series = ras_sharpe_adjustment(sharpe_ratios_array, complexity,
                                                              n_time_periods, n_strategies)
    adjusted_annual_sharpe_ratios: pd.Series = adjusted_sharpe_ratios * sqrt_n

    # Print adjusted Sharpe Ratios
    print(f"RAS-Adjusted Sharpe Ratios: {adjusted_annual_sharpe_ratios}")


if __name__ == '__main__':
    main()
