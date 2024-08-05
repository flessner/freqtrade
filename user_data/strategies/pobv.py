import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import IStrategy

pd.options.mode.chained_assignment = None
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# PrawnstarOBV
# source: https://github.com/davidzr/freqtrade-strategies/blob/9623c1f3d8c7f60c8b411010fa26377e6ca99ab9/strategies/PrawnstarOBV/PrawnstarOBV.py


class POBV(IStrategy):
    INTERFACE_VERSION = 2

    # Optimal timeframe for the strategy
    timeframe = "1h"

    minimal_roi = {"0": 0.1, "120": 0.05, "240": 0.025, "360": 0}

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.04
    # trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    use_buy_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        "buy": "limit",
        "sell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Momentum Indicators
        # ------------------------------------

        # Momentum
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obvSma"] = ta.SMA(dataframe["obv"], timeperiod=7)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["obv"], dataframe["obvSma"]))
                & (dataframe["rsi"] < 50)
                | (
                    (dataframe["obvSma"] - dataframe["close"]) / dataframe["obvSma"]
                    > 0.1
                )
                | (dataframe["obv"] > dataframe["obv"].shift(1))
                & (dataframe["obvSma"] > dataframe["obvSma"].shift(5))
                & (dataframe["rsi"] < 50)
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[(), "sell"] = 1

        return dataframe
