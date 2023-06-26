import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import yfinance as yf

# Acquire Market Data
# Load historical market data into a pandas DataFrame
def load_market_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Build a Backtesting Engine
# Implement the backtesting engine that simulates trades and calculates performance
class BacktestingEngine:
    def __init__(self, data):
        self.data = data
        self.portfolio = {'cash': 1000000, 'positions': {}}
        self.trades = []

    def run_backtest(self):
        risk_per_trade = 0.02  # Risk 2% of portfolio per trade

        for index, row in self.data.iterrows():
            # Implement your trading logic for each data point
            close_price = row['Close']
            signal = row['signal'] 

            if signal == 1:
                # Buy signal
                available_cash = self.portfolio['cash']
                position_size = available_cash * risk_per_trade / close_price
                self.execute_trade(symbol=self.data.name, quantity=position_size, price=close_price)
            elif signal == -1:
                # Sell signal
                position_size = self.portfolio['positions'].get(self.data.name, 0)
                if position_size > 0:
                    self.execute_trade(symbol=self.data.name, quantity=-position_size, price=close_price)


    def calculate_performance(self):
        # Calculate performance metrics based on executed trades
        trade_prices = np.array([trade['price'] for trade in self.trades])
        trade_quantities = np.array([trade['quantity'] for trade in self.trades])

        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]

        total_pnl = np.sum(trade_pnl)
        average_trade_return = np.mean(trade_returns)
        win_ratio = np.sum(trade_pnl > 0) / len(trade_pnl)

        return total_pnl, average_trade_return, win_ratio

    def execute_trade(self, symbol, quantity, price):
        # Update portfolio and execute trade
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity

        self.trades.append({'symbol': symbol, 'quantity': quantity, 'price': price})

    def get_portfolio_value(self, price):
        # Calculate the current value of the portfolio
        positions_value = sum(self.portfolio['positions'].get(symbol, 0) * price for symbol in self.portfolio['positions'])
        return self.portfolio['cash'] + positions_value

    def get_portfolio_returns(self):
        # Calculate the daily portfolio returns
        portfolio_value = [self.get_portfolio_value(row['Close']) for _, row in self.data.iterrows()]
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)

    def plot_portfolio_value(self):
        portfolio_value = [self.get_portfolio_value(row['Close']) for _, row in self.data.iterrows()]
        dates = self.data.index
        signals = self.data['signal']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot portfolio value
        ax1.plot(dates, portfolio_value, label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')

        # Plot buy/sell signals
        ax2 = ax1.twinx()
        ax2.plot(dates, signals, 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)

        fig.tight_layout()
        plt.show()


def main():
    print("Greetings! Enter the ticker for the security you'd like to explore:")
    # Implement Data Feed Integration
    # Load market data
    symbol = input("Ticker: ")
    start_date = input("Start Date (YYYY-MM-DD): ")
    end_date = input("End Date (YYYY-MM-DD): ")
    data = load_market_data(symbol, start_date, end_date)

    # Define Trading Strategies
    # Implement your trading strategies using the data
    print(data.columns)
    sma_short = ta.trend.sma_indicator(data['Close'], window=50)
    sma_long = ta.trend.sma_indicator(data['Close'], window=200)

    data['signal'] = np.where(sma_short > sma_long, 1, -1)
    data.name = symbol  # Set the name attribute for the data DataFrame

    engine = BacktestingEngine(data)
    initial_portfolio_value = engine.get_portfolio_value(data.iloc[0]['Close'])

    engine.run_backtest()


    # Step 8: Evaluate Performance
    final_portfolio_value = engine.get_portfolio_value(data.iloc[-1]['Close'])
    returns = engine.get_portfolio_returns()
    total_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    annualized_returns = (1 + total_returns) ** (252 / len(data)) - 1  # Assuming 252 trading days in a year
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (annualized_returns - 0.02) / volatility  # Assuming risk-free rate of 2%

    # Visualize Results
    engine.plot_portfolio_value()


    # Calculate and print performance metrics
    total_pnl, average_trade_return, win_ratio = engine.calculate_performance()
    print('--- Performance Metrics ---')
    print(f'Total Returns: {total_returns:.2%}')
    print(f'Annualized Returns: {annualized_returns:.2%}')
    print(f'Volatility: {volatility:.2%}')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(f'Total P&L: {total_pnl:.2f}')
    print(f'Average Trade Return: {average_trade_return:.2%}')
    print(f'Win Ratio: {win_ratio:.2%}')


if __name__ == '__main__':
    main()
