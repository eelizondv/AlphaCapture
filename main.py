## Data Collection
# requirements
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# example tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# download data
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

# calculate daily returns
returns = data.pct_change()

# additional technical indicators
for ticker in tickers:
    data[f'{ticker}_SMA_20'] = data[ticker].rolling(window=20).mean()
    data[f'{ticker}_SMA_50'] = data[ticker].rolling(window=50).mean()
    data[f'{ticker}_EMA_20'] = data[ticker].ewm(span=20, adjust=False).mean()
    data[f'{ticker}_Momentum'] = returns[ticker].rolling(window=20).mean()

# merge all indicators into a single df
indicators = pd.concat([data, returns.rename(columns={ticker: f'{ticker}_Returns' for ticker in tickers})], axis=1)
indicators.tail()


## Feature Engineering
# download assuming volume data is available
volume_data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Volume']

# calculate volatility and volume based indicators
for ticker in tickers:
    indicators[f'{ticker}_Volatility'] = returns[ticker].rolling(window=20).std()
    indicators[f'{ticker}_Volume'] = volume_data[ticker]
    indicators[f'{ticker}_Volume_SMA_20'] = volume_data[ticker].rolling(window=20).mean()
indicators.tail()


## ML for Signal Generation
# requirements
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# prep data for ML
features = indicators.drop(columns=['AAPL_Returns','GOOGL_Returns','AMZN_Returns','META_Returns','MSFT_Returns'])
target = (returns > 0).astype(int)  # Binary target: 1 if return is positive, else 0

# drow rows with NaN values
print(features.isnull().sum()) # check for NaN values
features = features.dropna()
target = target.reindex(features.index)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict signals
predictions = model.predict(X_test)

# evaluate the model
print(classification_report(y_test, predictions))


## Backtesting with Transaction Costs
# define transaction cost
transaction_cost = 0.001  # 0.1%

# generate signals
signals = pd.DataFrame(predictions, index=X_test.index, columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])

# calculate positions
positions = signals.shift(1)

# calculate portfolio returns considering transaction costs
portfolio_returns = positions * returns.loc[X_test.index] - transaction_cost * np.abs(positions.diff())

# calculate cumulative portfolio value
initial_capital = 100000
portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital

total_returns = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
for stock in total_returns.index:
    print(f"Total Returns ({stock}): {total_returns[stock]:.2%}")

# plot portfolio value over time
portfolio_value.plot()
plt.title('Portfolio Value Over Time (with Transaction Costs)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.show()


## Performance Metrics
# calculate Sharpe Ratio
portfolio_returns = portfolio_value.pct_change().dropna()
sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)

for asset in sharpe_ratio.index:
    print(f"Sharpe Ratio ({asset}): {sharpe_ratio[asset]:.2f}")

# calculate maximum drawdown
rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max
max_drawdown = drawdown.min()

for asset in max_drawdown.index:
    print(f"Max Drawdown ({asset}): {max_drawdown[asset]:.2%}")

# plot drawdown
drawdown.plot()
plt.title('Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.show()


## Optimization and Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Grid search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")


## Risk Management
# Define stop-loss and take-profit thresholds
stop_loss = 0.02  # 2%
take_profit = 0.05  # 5%

# Initialize variables
capital = initial_capital
positions = {ticker: 0 for ticker in tickers}
portfolio_value = []

# Reindex signals to match data index
signals = signals.reindex(data.index, method='pad').fillna(0)

for i in range(1, len(signals)):
    current_date = signals.index[i]
    current_prices = data.loc[current_date]
    
    for ticker in tickers:
        if positions[ticker] == 0:  # No current position
            if signals[ticker].iloc[i] == 1:
                positions[ticker] = capital / current_prices[ticker]
                buy_price = current_prices[ticker]
        else:  # Current position exists
            if current_prices[ticker] >= buy_price * (1 + take_profit) or current_prices[ticker] <= buy_price * (1 - stop_loss):
                capital = positions[ticker] * current_prices[ticker] - transaction_cost * positions[ticker] * current_prices[ticker]
                positions[ticker] = 0
    
    portfolio_value.append(capital + sum(positions[ticker] * current_prices[ticker] for ticker in tickers))

# Convert to Series for easier handling
portfolio_value = pd.Series(portfolio_value, index=data.index[1:])

print(f"Total Returns: {(portfolio_value.iloc[-1] - initial_capital) / initial_capital:.2%}")

# Plot portfolio value over time with risk management
start_date = '2022-05-01'
portfolio_value[start_date:].plot()
plt.title('Portfolio Value Over Time (with Risk Management)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.show()