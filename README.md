# Rademacher Anti-Serum for Strategy Performance Adjustment

This repository provides a Python implementation of the Rademacher Anti-Serum, based on the work of Giuseppe A. Paleologo. This approach uses Rademacher Complexity to adjust the Sharpe Ratios of strategies, helping to correct for overfitting biases and providing a more reliable assessment of strategy performance.

## Features
- **Rademacher Complexity Estimation**: Computes the empirical Rademacher Complexity for a set of strategy returns.
- **RAS-Adjusted Sharpe Ratios**: Adjusts Sharpe Ratios based on the Rademacher Complexity, yielding a more conservative estimate of expected performance.

## Background
Rademacher Complexity measures how well a set of strategies aligns with random noise. High Rademacher Complexity indicates overfitting, as it suggests that a strategy's performance could be due to random fluctuations. The RAS adjustment corrects for this bias by adjusting Sharpe Ratios, producing a more robust performance estimate.

## Installation
This code requires Python 3.8 or higher. You can install the necessary libraries via:
```
pip install numpy pandas
```

## Usage
1. Prepare Your Data: Organize strategy returns in a DataFrame, with each column representing a strategy.
2. Calculate Rademacher Complexity:
```
complexity = rademacher_complexity(returns_matrix)
```
3. RAS-Adjusted Sharpe Ratios:
```
adjusted_sharpes = ras_sharpe_adjustment(sharpe_ratios, complexity, shape_t, shape_n)
```
## Example
Assuming 'returns_df' is a DataFrame with strategy returns
```
complexity = rademacher_complexity(returns_df.values)
adjusted_sharpe_ratios = ras_sharpe_adjustment(sharpe_ratios, complexity, returns_df.shape[0], returns_df.shape[1])
print(adjusted_sharpe_ratios)
```
## References
Paleologo, G. A. (2024). The Elements of Quantitative Investing (Draft)