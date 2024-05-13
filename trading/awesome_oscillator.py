def awesome_oscillator(high, low, short_span=5, long_span=34):
    """Calculate the Awesome Oscillator.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        short_span (int): Short window period.
        long_span (int): Long window period.

    Returns:
        pd.Series: The Awesome Oscillator values.
    """
    midpoint = (high + low) / 2
    sma_short = midpoint.rolling(window=short_span).mean()
    sma_long = midpoint.rolling(window=long_span).mean()

    ao = sma_short - sma_long
    return ao


def ao_scalping_strategy(data, short_span=5, long_span=34):
    """Scalping strategy based on the Awesome Oscillator.
    Args:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' price columns.
        short_span (int): Short window period for AO.
        long_span (int): Long window period for AO.

    Returns:
        pd.DataFrame: DataFrame with signals and AO values.
    """
    ao_values = awesome_oscillator(data['High'], data['Low'], short_span, long_span)
    data['AO'] = ao_values
    data['Signal'] = 0
    data.loc[data['AO'] > 0, 'Signal'] = 1
    data.loc[data['AO'] < 0, 'Signal'] = -1

    # Detect crossovers
    data['Position'] = data['Signal'].diff()

    return data[['AO', 'Signal', 'Position']]


# Example data
data = pd.DataFrame({
    'High': [120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
    'Low': [115, 116, 117, 118, 119, 120, 121, 122, 123, 124]
})

results = ao_scalping_strategy(data)
print(results)
