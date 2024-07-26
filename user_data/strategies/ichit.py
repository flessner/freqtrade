from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import technical.indicators as ftt
from functools import reduce
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

pd.options.mode.chained_assignment = None

# ichiV1 - f(ast) variant of the original ichi strategy
# source: https://github.com/PeetCrypto/freqtrade-stuff/blob/main/IchisV1.py

leverage = 10


class IchiT(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_fan_magnitude_shift_value": 4,
        "buy_min_fan_magnitude_gain": 1.00581,
        "buy_trend_above_senkou_level": 2,
        "buy_trend_bullish_level": 1,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_trend_indicator": "trend_close_15m",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.04 * leverage,
        "10": 0.025 * leverage,
        "25": 0.01 * leverage,
        "50": 0,
    }

    # Stoploss:
    stoploss = -0.0125 * leverage

    # Optimal timeframe for the strategy
    timeframe = "5m"

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False
    # trailing_stop_positive = 0.002
    # trailing_stop_positive_offset = 0.025
    # trailing_only_offset_is_reached = True

    # more variables
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # hyperoptable parameters
    buy_trend_above_senkou_level = IntParameter(
        1,
        8,
        default=buy_params["buy_trend_above_senkou_level"],
        space="buy",
        optimize=True,
    )
    buy_trend_bullish_level = IntParameter(
        1, 8, default=buy_params["buy_trend_bullish_level"], space="buy", optimize=True
    )
    buy_fan_magnitude_shift_value = IntParameter(
        1,
        5,
        default=buy_params["buy_fan_magnitude_shift_value"],
        space="buy",
        optimize=True,
    )
    buy_min_fan_magnitude_gain = DecimalParameter(
        1,
        1.05,
        default=buy_params["buy_min_fan_magnitude_gain"],
        decimals=5,
        space="buy",
        optimize=True,
    )

    sell_trend_indicator = CategoricalParameter(
        [
            "trend_close_15m",
            "trend_close_30m",
            "trend_close_1h",
            "trend_close_2h",
            "trend_close_4h",
            "trend_close_6h",
            "trend_close_8h",
        ],
        default=sell_params["sell_trend_indicator"],
        space="sell",
        optimize=True,
    )

    def leverage(self, **kwargs) -> float:
        return leverage

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["open"] = heikinashi["open"]
        # dataframe['close'] = heikinashi['close']
        dataframe["high"] = heikinashi["high"]
        dataframe["low"] = heikinashi["low"]

        dataframe["trend_close_5m"] = dataframe["close"]
        dataframe["trend_close_15m"] = ta.EMA(dataframe["close"], timeperiod=3)
        dataframe["trend_close_30m"] = ta.EMA(dataframe["close"], timeperiod=6)
        dataframe["trend_close_1h"] = ta.EMA(dataframe["close"], timeperiod=12)
        dataframe["trend_close_2h"] = ta.EMA(dataframe["close"], timeperiod=24)
        dataframe["trend_close_4h"] = ta.EMA(dataframe["close"], timeperiod=48)
        dataframe["trend_close_6h"] = ta.EMA(dataframe["close"], timeperiod=72)
        dataframe["trend_close_8h"] = ta.EMA(dataframe["close"], timeperiod=96)

        dataframe["trend_open_5m"] = dataframe["open"]
        dataframe["trend_open_15m"] = ta.EMA(dataframe["open"], timeperiod=3)
        dataframe["trend_open_30m"] = ta.EMA(dataframe["open"], timeperiod=6)
        dataframe["trend_open_1h"] = ta.EMA(dataframe["open"], timeperiod=12)
        dataframe["trend_open_2h"] = ta.EMA(dataframe["open"], timeperiod=24)
        dataframe["trend_open_4h"] = ta.EMA(dataframe["open"], timeperiod=48)
        dataframe["trend_open_6h"] = ta.EMA(dataframe["open"], timeperiod=72)
        dataframe["trend_open_8h"] = ta.EMA(dataframe["open"], timeperiod=96)

        dataframe["fan_magnitude"] = (
            dataframe["trend_close_1h"] / dataframe["trend_close_8h"]
        )
        dataframe["fan_magnitude_gain"] = dataframe["fan_magnitude"] / dataframe[
            "fan_magnitude"
        ].shift(1)

        ichimoku = ftt.ichimoku(
            dataframe,
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=30,
        )
        dataframe["chikou_span"] = ichimoku["chikou_span"]
        dataframe["tenkan_sen"] = ichimoku["tenkan_sen"]
        dataframe["kijun_sen"] = ichimoku["kijun_sen"]
        dataframe["senkou_a"] = ichimoku["senkou_span_a"]
        dataframe["senkou_b"] = ichimoku["senkou_span_b"]
        dataframe["leading_senkou_span_a"] = ichimoku["leading_senkou_span_a"]
        dataframe["leading_senkou_span_b"] = ichimoku["leading_senkou_span_b"]
        dataframe["cloud_green"] = ichimoku["cloud_green"]
        dataframe["cloud_red"] = ichimoku["cloud_red"]

        dataframe["atr"] = ta.ATR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Trending market
        if self.buy_trend_above_senkou_level.value >= 1:
            conditions.append(dataframe["trend_close_5m"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_5m"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 2:
            conditions.append(dataframe["trend_close_15m"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_15m"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 3:
            conditions.append(dataframe["trend_close_30m"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_30m"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 4:
            conditions.append(dataframe["trend_close_1h"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_1h"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 5:
            conditions.append(dataframe["trend_close_2h"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_2h"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 6:
            conditions.append(dataframe["trend_close_4h"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_4h"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 7:
            conditions.append(dataframe["trend_close_6h"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_6h"] > dataframe["senkou_b"])

        if self.buy_trend_above_senkou_level.value >= 8:
            conditions.append(dataframe["trend_close_8h"] > dataframe["senkou_a"])
            conditions.append(dataframe["trend_close_8h"] > dataframe["senkou_b"])

        # Trends bullish
        if self.buy_trend_bullish_level.value >= 1:
            conditions.append(dataframe["trend_close_5m"] > dataframe["trend_open_5m"])

        if self.buy_trend_bullish_level.value >= 2:
            conditions.append(
                dataframe["trend_close_15m"] > dataframe["trend_open_15m"]
            )

        if self.buy_trend_bullish_level.value >= 3:
            conditions.append(
                dataframe["trend_close_30m"] > dataframe["trend_open_30m"]
            )

        if self.buy_trend_bullish_level.value >= 4:
            conditions.append(dataframe["trend_close_1h"] > dataframe["trend_open_1h"])

        if self.buy_trend_bullish_level.value >= 5:
            conditions.append(dataframe["trend_close_2h"] > dataframe["trend_open_2h"])

        if self.buy_trend_bullish_level.value >= 6:
            conditions.append(dataframe["trend_close_4h"] > dataframe["trend_open_4h"])

        if self.buy_trend_bullish_level.value >= 7:
            conditions.append(dataframe["trend_close_6h"] > dataframe["trend_open_6h"])

        if self.buy_trend_bullish_level.value >= 8:
            conditions.append(dataframe["trend_close_8h"] > dataframe["trend_open_8h"])

        # Trends magnitude
        conditions.append(
            dataframe["fan_magnitude_gain"] >= self.buy_min_fan_magnitude_gain.value
        )
        conditions.append(dataframe["fan_magnitude"] > 1)

        for x in range(self.buy_fan_magnitude_shift_value.value):
            conditions.append(
                dataframe["fan_magnitude"].shift(x + 1) < dataframe["fan_magnitude"]
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
