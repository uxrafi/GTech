"""
test_manual_strategy.py

This test suite validates the functionality of the ManualStrategy class 
from the ML4T project.

It verifies that:
- The strategy produces a trades DataFrame with expected structure and valid trades.
- The trades cover the specified date range appropriately, accounting for indicator warm-up.
- The final position is properly closed at the end of the trading period.
- The strategy handles edge cases such as single-day date ranges.
- A ValueError is raised when no price data is available (e.g., invalid symbol or date range).

These tests ensure that ManualStrategy behaves as expected under typical and edge scenarios,
and that it complies with the API contract for use in the ML4T trading framework.
"""

import unittest
import datetime as dt
import pandas as pd
from ManualStrategy import ManualStrategy

class TestManualStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = ManualStrategy(verbose=False)
        self.symbol = "JPM"
        self.sv = 100000
        self.sd = dt.datetime(2008, 1, 1)
        self.ed = dt.datetime(2009, 12, 31)

    def test_trade_output(self):
        """Test that trades DataFrame has expected structure and some trades occurred."""
        trades = self.strategy.testPolicy(symbol=self.symbol, sd=self.sd, ed=self.ed, sv=self.sv)
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIn(self.symbol, trades.columns)
        self.assertTrue((trades.values == 0).sum() < trades.size)  # Some trades occurred
        self.assertTrue(trades.index.min() >= self.sd)
        self.assertEqual(trades.index.max(), self.ed)

    def test_no_data_error(self):
        """Test that querying an invalid symbol/date range raises ValueError."""
        bad_symbol = "BADSYMBOL"
        bad_sd = dt.datetime(1900, 1, 1)
        bad_ed = dt.datetime(1900, 12, 31)
        with self.assertRaises(ValueError):
            self.strategy.testPolicy(symbol=bad_symbol, sd=bad_sd, ed=bad_ed, sv=self.sv)

    def test_final_position_closed(self):
        """Test that the final day's trades close out any open positions."""
        trades = self.strategy.testPolicy(symbol=self.symbol, sd=self.sd, ed=self.ed, sv=self.sv)
        final_day_trades = trades.iloc[-1][self.symbol]
        self.assertIn(final_day_trades, [-1000, 0, 1000, -2000, 2000])  # Final trade should close or no trade
        net_position = trades[self.symbol].sum()  # Sum of all trades should be zero to close position
        self.assertEqual(net_position, 0)

    def test_single_day_range(self):
        """Test that the method works without error for a single day range."""
        single_day = dt.datetime(2008, 1, 15)
        trades = self.strategy.testPolicy(symbol=self.symbol, sd=single_day, ed=single_day, sv=self.sv)
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertEqual(len(trades), 1)

    def test_short_range(self):
        """Test method behavior on a short multi-day range."""
        short_sd = dt.datetime(2008, 1, 2)
        short_ed = dt.datetime(2008, 1, 10)
        trades = self.strategy.testPolicy(symbol=self.symbol, sd=short_sd, ed=short_ed, sv=self.sv)
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertTrue(len(trades) >= 1)

if __name__ == "__main__":
    unittest.main()

