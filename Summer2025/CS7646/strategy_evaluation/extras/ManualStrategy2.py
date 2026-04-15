import numpy as np
import pandas as pd
import datetime as dt
from util import get_data
from indicators import sma, ema, momentum, bollinger_bands, macd

class ManualStrategy:
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Manual strategy using technical indicators.
        Parameters match project requirements.
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        
        # Indicator parameters (same as StrategyLearner)
        self.sma_window = 20
        self.ema_span = 10
        self.momentum_window = 10
        self.bb_window = 20
        self.macd_short = 12
        self.macd_long = 26

    def author(self):
        return "urafi3"

    def study_group(self):
        # Return comma-separated GT usernames in your study group; for solo:
        return "urafi3"

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
        """
        Required method for API compliance.
        Manual strategy does not need training, so this is a no-op.
        """
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        """Generate trades based on manual rules"""
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]]
        
        # Calculate indicators
        sma_val = sma(prices, window=self.sma_window)
        ema_val = ema(prices, span=self.ema_span)
        momentum_val = momentum(prices, window=self.momentum_window)
        bb_val, bb_upper, bb_lower = bollinger_bands(prices, window=self.bb_window)[:3]
        macd_val, macd_signal = macd(prices, span_short=self.macd_short, span_long=self.macd_long)
        
        # Combine indicators
        indicators = pd.DataFrame({
            'Price': prices[symbol],
            'SMA': sma_val,
            'EMA': ema_val,
            'Momentum': momentum_val,
            'BB': bb_val,
            'MACD': macd_val,
            'MACD_Signal': macd_signal,
            'BB_Upper': bb_upper,
            'BB_Lower': bb_lower
        }).dropna()
        
        # Initialize trades DataFrame
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        
        # Track position and entry points
        position = 0
        
        # Manual trading rules
        for i in range(1, len(indicators)):
            price = indicators.iloc[i]['Price']
            sma = indicators.iloc[i]['SMA']
            ema = indicators.iloc[i]['EMA']
            momentum = indicators.iloc[i]['Momentum']
            bb = indicators.iloc[i]['BB']
            macd = indicators.iloc[i]['MACD'] - indicators.iloc[i]['MACD_Signal']
            
            # Entry signals (must use at least 3 indicators)
            if position == 0:
                # Long entry: Price below lower BB, negative momentum reverting, MACD crossing up
                if (price < indicators.iloc[i]['BB_Lower'] and 
                    momentum > -0.1 and 
                    macd > 0):
                    trades.loc[indicators.index[i], symbol] = 1000 - position
                    position = 1000
                
                # Short entry: Price above upper BB, high momentum, MACD crossing down
                elif (price > indicators.iloc[i]['BB_Upper'] and 
                      momentum > 0.2 and 
                      macd < 0):
                    trades.loc[indicators.index[i], symbol] = -1000 - position
                    position = -1000
            
            # Exit signals
            elif position > 0:  # Exit long
                if (price > sma * 1.02 or 
                    momentum < -0.05 or 
                    macd < 0):
                    trades.loc[indicators.index[i], symbol] = -position
                    position = 0
            elif position < 0:  # Exit short
                if (price < sma * 0.98 or 
                    momentum > 0.05 or 
                    macd > 0):
                    trades.loc[indicators.index[i], symbol] = -position
                    position = 0
        
        return trades
