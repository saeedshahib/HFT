import pandas as pd
import numpy as np
import unittest


def bollinger_bands(data, window=20, num_of_std=2):
    """Calculate Bollinger Bands.
    Args:
        data (pd.Series): Pandas Series of prices.
        window (int): Rolling window size.
        num_of_std (int): Number of standard deviations from the mean.

    Returns:
        pd.DataFrame: DataFrame containing the middle, upper, and lower bands.
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return pd.DataFrame({'Middle Band': rolling_mean, 'Upper Band': upper_band, 'Lower Band': lower_band})


class TestBollingerBands(unittest.TestCase):
    def test_bollinger_bands(self):
        data = pd.Series([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
        result = bollinger_bands(data, window=5, num_of_std=2)

        # Manually compute expected values for a specific point if needed
        expected_middle = data.rolling(window=5).mean().iloc[-1]
        expected_std = data.rolling(window=5).std().iloc[-1]
        expected_upper = expected_middle + (expected_std * 2)
        expected_lower = expected_middle - (expected_std * 2)

        # Check the last row of the output
        self.assertAlmostEqual(result['Middle Band'].iloc[-1], expected_middle)
        self.assertAlmostEqual(result['Upper Band'].iloc[-1], expected_upper)
        self.assertAlmostEqual(result['Lower Band'].iloc[-1], expected_lower)


# Run the tests
if __name__ == '__main__':
    unittest.main()


#  https://www.tradingview.com/script/QYCToBoN-Bollinger-Awesome-Alert-R1-by-JustUncleL/
