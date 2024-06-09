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

print(f"Total Returns: {(portfolio_value.iloc[-1].item() - initial_capital) / initial_capital:.2%}")

# plot portfolio value over time
portfolio_value.plot()
plt.title('Portfolio Value Over Time (with Transaction Costs)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.show()
