from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter

# SMAOffset
# source: https://github.com/davidzr/freqtrade-strategies/blob/main/strategies/SMAOffset/SMAOffset.py

ma_types = {
    "SMA": ta.SMA,
    "EMA": ta.EMA,
}


class SMAO(IStrategy):
    INTERFACE_VERSION = 2

    # hyperopt and paste results here
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 30,
        "buy_trigger": "SMA",
        "low_offset": 0.958,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 30,
        "high_offset": 1.012,
        "sell_trigger": "EMA",
    }

    # Stoploss:
    stoploss = -0.1

    # ROI table:
    minimal_roi = {
        "0": 1,
    }

    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params["base_nb_candles_buy"], space="buy"
    )
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params["base_nb_candles_sell"], space="sell"
    )
    low_offset = DecimalParameter(
        0.8, 0.99, default=buy_params["low_offset"], space="buy"
    )
    high_offset = DecimalParameter(
        0.8, 1.1, default=sell_params["high_offset"], space="sell"
    )
    buy_trigger = CategoricalParameter(
        ma_types.keys(), default=buy_params["buy_trigger"], space="buy"
    )
    sell_trigger = CategoricalParameter(
        ma_types.keys(), default=sell_params["sell_trigger"], space="sell"
    )

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.0001
    # trailing_stop_positive_offset = 0
    # trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy
    timeframe = "5m"

    use_exit_signal = True
    exit_profit_only = False

    process_only_new_candles = True
    startup_candle_count = 30

    plot_config = {
        "main_plot": {
            "ma_offset_buy": {"color": "orange"},
            "ma_offset_sell": {"color": "orange"},
        },
    }

    use_custom_stoploss = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.config["runmode"].value == "hyperopt":
            dataframe["ma_offset_buy"] = (
                ma_types[self.buy_trigger.value](
                    dataframe, int(self.base_nb_candles_buy.value)
                )
                * self.low_offset.value
            )
            dataframe["ma_offset_sell"] = (
                ma_types[self.sell_trigger.value](
                    dataframe, int(self.base_nb_candles_sell.value)
                )
                * self.high_offset.value
            )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config["runmode"].value == "hyperopt":
            dataframe["ma_offset_buy"] = (
                ma_types[self.buy_trigger.value](
                    dataframe, int(self.base_nb_candles_buy.value)
                )
                * self.low_offset.value
            )

        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ma_offset_buy"])
                & (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config["runmode"].value == "hyperopt":
            dataframe["ma_offset_sell"] = (
                ma_types[self.sell_trigger.value](
                    dataframe, int(self.base_nb_candles_sell.value)
                )
                * self.high_offset.value
            )

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ma_offset_sell"])
                & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1
        return dataframe
