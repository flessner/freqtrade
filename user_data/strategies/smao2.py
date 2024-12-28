from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta

# SMAOffset2
# source: https://github.com/davidzr/freqtrade-strategies/blob/main/strategies/SMAOffsetV2/SMAOffsetV2.py


class SMAO2(IStrategy):
    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.1
    timeframe = "5m"
    informative_timeframe = "1h"
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True

    use_custom_stoploss = False
    startup_candle_count = 200

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    @staticmethod
    def get_informative_indicators(dataframe: DataFrame, metadata: dict):
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=25)

        dataframe["go_long"] = ((dataframe["ema_fast"] > dataframe["ema_slow"])).astype(
            "int"
        ) * 2

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe

        informative = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )

        informative = self.get_informative_indicators(informative.copy(), metadata)

        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )
        # don't overwrite the base dataframe's HLCV information
        skip_columns = [
            (s + "_" + self.informative_timeframe)
            for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        dataframe.rename(
            columns=lambda s: (
                s.replace("_{}".format(self.informative_timeframe), "")
                if (not s in skip_columns)
                else s
            ),
            inplace=True,
        )

        sma_offset = 1 - 0.04
        sma_offset_pos = 1 + 0.012
        base_nb_candles = 20

        dataframe["sma_30_offset"] = (
            ta.SMA(dataframe, timeperiod=base_nb_candles) * sma_offset
        )
        dataframe["sma_30_offset_pos"] = (
            ta.SMA(dataframe, timeperiod=base_nb_candles) * sma_offset_pos
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["go_long"] > 0)
                & (dataframe["close"] < dataframe["sma_30_offset"])
                & (dataframe["volume"] > 0)
            ),
            "buy",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["go_long"] == 0)
                    | (dataframe["close"] > dataframe["sma_30_offset_pos"])
                )
                & (dataframe["volume"] > 0)
            ),
            "sell",
        ] = 1
        return dataframe
