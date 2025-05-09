import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
)
from pandas import DataFrame
from functools import reduce

# NFI5MO
# source:

leverage = 1


class NFI5MO(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.04 * leverage,
        "10": 0.025 * leverage,
        "25": 0.01 * leverage,
        "50": 0,
    }

    # Stoploss:
    stoploss = -0.01

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.166
    # trailing_stop_positive_offset = 0.263
    # trailing_only_offset_is_reached = True

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 32,
        "buy_bb20_close_bblowerband_4": 0.964,
        "buy_bb20_volume_4": 17.42,
        "buy_bb40_bbdelta_close_3": 0.025,
        "buy_bb40_closedelta_close_3": 0.025,
        "buy_bb40_tail_bbdelta_3": 0.228,
        "buy_bb_offset_10": 0.971,
        "buy_bb_offset_14": 0.987,
        "buy_bb_offset_18": 0.986,
        "buy_bb_offset_2": 0.991,
        "buy_bb_offset_5": 0.999,
        "buy_bb_offset_6": 0.999,
        "buy_bb_offset_9": 0.978,
        "buy_chop_min_19": 43.3,
        "buy_condition_10_enable": False,
        "buy_condition_11_enable": False,
        "buy_condition_12_enable": True,
        "buy_condition_13_enable": True,
        "buy_condition_14_enable": True,
        "buy_condition_15_enable": False,
        "buy_condition_16_enable": False,
        "buy_condition_17_enable": True,
        "buy_condition_18_enable": False,
        "buy_condition_19_enable": False,
        "buy_condition_1_enable": True,
        "buy_condition_20_enable": False,
        "buy_condition_21_enable": False,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": False,
        "buy_condition_4_enable": False,
        "buy_condition_5_enable": False,
        "buy_condition_6_enable": False,
        "buy_condition_7_enable": True,
        "buy_condition_8_enable": False,
        "buy_condition_9_enable": True,
        "buy_dip_threshold_1": 0.048,
        "buy_dip_threshold_10": 0.13,
        "buy_dip_threshold_11": 0.105,
        "buy_dip_threshold_12": 0.412,
        "buy_dip_threshold_2": 0.112,
        "buy_dip_threshold_3": 0.378,
        "buy_dip_threshold_4": 0.48,
        "buy_dip_threshold_5": 0.044,
        "buy_dip_threshold_6": 0.037,
        "buy_dip_threshold_7": 0.325,
        "buy_dip_threshold_8": 0.476,
        "buy_dip_threshold_9": 0.045,
        "buy_ema_open_mult_14": 0.026,
        "buy_ema_open_mult_15": 0.024,
        "buy_ema_open_mult_5": 0.02,
        "buy_ema_open_mult_6": 0.03,
        "buy_ema_open_mult_7": 0.032,
        "buy_ema_rel_15": 0.997,
        "buy_ema_rel_3": 0.98,
        "buy_ema_rel_5": 0.998,
        "buy_ema_rel_7": 0.979,
        "buy_ewo_12": 3.8,
        "buy_ewo_13": -13.5,
        "buy_ewo_16": 7.8,
        "buy_ewo_17": -14.4,
        "buy_ma_offset_10": 0.945,
        "buy_ma_offset_11": 0.985,
        "buy_ma_offset_12": 0.961,
        "buy_ma_offset_13": 0.971,
        "buy_ma_offset_14": 0.981,
        "buy_ma_offset_15": 0.971,
        "buy_ma_offset_16": 0.935,
        "buy_ma_offset_17": 0.955,
        "buy_ma_offset_9": 0.971,
        "buy_mfi_1": 37.8,
        "buy_mfi_11": 53.9,
        "buy_mfi_2": 42.6,
        "buy_mfi_9": 51.5,
        "buy_min_inc_1": 0.039,
        "buy_min_inc_11": 0.013,
        "buy_pump_pull_threshold_1": 1.81,
        "buy_pump_pull_threshold_2": 2.39,
        "buy_pump_pull_threshold_3": 2.6,
        "buy_pump_pull_threshold_4": 2.7,
        "buy_pump_pull_threshold_5": 2.77,
        "buy_pump_pull_threshold_6": 2.21,
        "buy_pump_pull_threshold_7": 2.17,
        "buy_pump_pull_threshold_8": 2.88,
        "buy_pump_pull_threshold_9": 2.71,
        "buy_pump_threshold_1": 0.968,
        "buy_pump_threshold_2": 0.405,
        "buy_pump_threshold_3": 0.82,
        "buy_pump_threshold_4": 0.978,
        "buy_pump_threshold_5": 0.683,
        "buy_pump_threshold_6": 0.754,
        "buy_pump_threshold_7": 0.42,
        "buy_pump_threshold_8": 0.469,
        "buy_pump_threshold_9": 1.592,
        "buy_rsi_1": 37.8,
        "buy_rsi_11": 47.7,
        "buy_rsi_12": 38.2,
        "buy_rsi_15": 48.4,
        "buy_rsi_16": 26.3,
        "buy_rsi_18": 22.0,
        "buy_rsi_1h_10": 20.0,
        "buy_rsi_1h_20": 21.7,
        "buy_rsi_1h_21": 37.8,
        "buy_rsi_1h_diff_2": 43.1,
        "buy_rsi_1h_max_1": 84.6,
        "buy_rsi_1h_max_11": 85.6,
        "buy_rsi_1h_max_2": 90.3,
        "buy_rsi_1h_max_9": 79.0,
        "buy_rsi_1h_min_1": 39.4,
        "buy_rsi_1h_min_11": 45.7,
        "buy_rsi_1h_min_19": 57.4,
        "buy_rsi_1h_min_2": 37.2,
        "buy_rsi_1h_min_9": 39.5,
        "buy_rsi_20": 28.5,
        "buy_rsi_21": 17.8,
        "buy_rsi_7": 39.5,
        "buy_rsi_8": 39.6,
        "buy_tail_diff_8": 5.7,
        "buy_volume_10": 5.6,
        "buy_volume_12": 4.5,
        "buy_volume_13": 2.3,
        "buy_volume_14": 7.4,
        "buy_volume_15": 7.1,
        "buy_volume_16": 7.0,
        "buy_volume_17": 3.4,
        "buy_volume_18": 1.4,
        "buy_volume_2": 4.1,
        "buy_volume_20": 5.2,
        "buy_volume_21": 3.9,
        "buy_volume_7": 1.1,
        "buy_volume_8": 3.8,
        "buy_volume_9": 3.95,
        "ewo_high": 6.899,
        "ewo_low": -15.271,
        "fast_ewo": 46,
        "low_offset_ema": 0.968,
        "low_offset_kama": 0.934,
        "low_offset_sma": 0.969,
        "low_offset_t3": 0.938,
        "low_offset_trima": 0.973,
        "slow_ewo": 157,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset_ema": 1.012,
        "high_offset_kama": 1.012,
        "high_offset_sma": 1.012,
        "high_offset_t3": 1.012,
        "high_offset_trima": 1.012,
    }

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = "5m"
    inf_1h = "1h"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 300

    #############################################################

    # Multi Offset
    base_nb_candles_buy = IntParameter(
        5, 80, default=20, load=True, space="buy", optimize=True
    )
    base_nb_candles_sell = IntParameter(
        5, 80, default=20, load=True, space="sell", optimize=True
    )
    low_offset_sma = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space="buy", optimize=True
    )
    high_offset_sma = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space="sell", optimize=True
    )
    low_offset_ema = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space="buy", optimize=True
    )
    high_offset_ema = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space="sell", optimize=True
    )
    low_offset_trima = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space="buy", optimize=True
    )
    high_offset_trima = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space="sell", optimize=True
    )
    low_offset_t3 = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space="buy", optimize=True
    )
    high_offset_t3 = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space="sell", optimize=True
    )
    low_offset_kama = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space="buy", optimize=True
    )
    high_offset_kama = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space="sell", optimize=True
    )

    # MA list
    ma_types = ["sma", "ema", "trima", "t3", "kama"]
    ma_map = {
        "sma": {
            "low_offset": low_offset_sma.value,
            "high_offset": high_offset_sma.value,
            "calculate": ta.SMA,
        },
        "ema": {
            "low_offset": low_offset_ema.value,
            "high_offset": high_offset_ema.value,
            "calculate": ta.EMA,
        },
        "trima": {
            "low_offset": low_offset_trima.value,
            "high_offset": high_offset_trima.value,
            "calculate": ta.TRIMA,
        },
        "t3": {
            "low_offset": low_offset_t3.value,
            "high_offset": high_offset_t3.value,
            "calculate": ta.T3,
        },
        "kama": {
            "low_offset": low_offset_kama.value,
            "high_offset": high_offset_kama.value,
            "calculate": ta.KAMA,
        },
    }

    # Protection
    ewo_low = DecimalParameter(
        -20.0,
        -8.0,
        default=buy_params["ewo_low"],
        load=True,
        space="buy",
        optimize=True,
    )
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params["ewo_high"], load=True, space="buy", optimize=True
    )
    fast_ewo = IntParameter(
        10, 50, default=buy_params["fast_ewo"], load=True, space="buy", optimize=True
    )
    slow_ewo = IntParameter(
        100, 200, default=buy_params["slow_ewo"], load=True, space="buy", optimize=True
    )

    buy_condition_1_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_1_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_2_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_2_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_3_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_3_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_4_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_4_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_5_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_5_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_6_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_6_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_7_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_7_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_8_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_8_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_9_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_9_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_10_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_10_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_11_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_11_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_12_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_12_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_13_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_13_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_14_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_14_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_15_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_15_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_16_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_16_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_17_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_17_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_18_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_18_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_19_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_19_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_20_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_20_enable"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_condition_21_enable = CategoricalParameter(
        [True, False],
        default=buy_params["buy_condition_21_enable"],
        space="buy",
        optimize=True,
        load=True,
    )

    # Normal dips
    buy_dip_threshold_1 = DecimalParameter(
        0.001,
        0.05,
        default=buy_params["buy_dip_threshold_1"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_2 = DecimalParameter(
        0.01,
        0.2,
        default=buy_params["buy_dip_threshold_2"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_3 = DecimalParameter(
        0.05,
        0.4,
        default=buy_params["buy_dip_threshold_3"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_4 = DecimalParameter(
        0.2,
        0.5,
        default=buy_params["buy_dip_threshold_4"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # Strict dips
    buy_dip_threshold_5 = DecimalParameter(
        0.001,
        0.05,
        default=buy_params["buy_dip_threshold_5"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_6 = DecimalParameter(
        0.01,
        0.2,
        default=buy_params["buy_dip_threshold_6"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_7 = DecimalParameter(
        0.05,
        0.4,
        default=buy_params["buy_dip_threshold_7"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_8 = DecimalParameter(
        0.2,
        0.5,
        default=buy_params["buy_dip_threshold_8"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # Loose dips
    buy_dip_threshold_9 = DecimalParameter(
        0.001,
        0.05,
        default=buy_params["buy_dip_threshold_9"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_10 = DecimalParameter(
        0.01,
        0.2,
        default=buy_params["buy_dip_threshold_10"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_11 = DecimalParameter(
        0.05,
        0.4,
        default=buy_params["buy_dip_threshold_11"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_dip_threshold_12 = DecimalParameter(
        0.2,
        0.5,
        default=buy_params["buy_dip_threshold_12"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    # 24 hours
    buy_pump_pull_threshold_1 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_1"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_1 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_1"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 36 hours
    buy_pump_pull_threshold_2 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_2"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_2 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_2"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 48 hours
    buy_pump_pull_threshold_3 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_3"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_3 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_3"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    # 24 hours strict
    buy_pump_pull_threshold_4 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_4"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_4 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_4"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 36 hours strict
    buy_pump_pull_threshold_5 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_5"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_5 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_5"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 48 hours strict
    buy_pump_pull_threshold_6 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_6"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_6 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_6"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    # 24 hours loose
    buy_pump_pull_threshold_7 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_7"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_7 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_7"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 36 hours loose
    buy_pump_pull_threshold_8 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_8"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_8 = DecimalParameter(
        0.4,
        1.0,
        default=buy_params["buy_pump_threshold_8"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    # 48 hours loose
    buy_pump_pull_threshold_9 = DecimalParameter(
        1.5,
        3.0,
        default=buy_params["buy_pump_pull_threshold_9"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_pump_threshold_9 = DecimalParameter(
        0.4,
        1.8,
        default=buy_params["buy_pump_threshold_9"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_min_inc_1 = DecimalParameter(
        0.01,
        0.05,
        default=buy_params["buy_min_inc_1"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_min_1 = DecimalParameter(
        25.0,
        40.0,
        default=buy_params["buy_rsi_1h_min_1"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_max_1 = DecimalParameter(
        70.0,
        90.0,
        default=buy_params["buy_rsi_1h_max_1"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1 = DecimalParameter(
        20.0,
        40.0,
        default=buy_params["buy_rsi_1"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_mfi_1 = DecimalParameter(
        20.0,
        40.0,
        default=buy_params["buy_mfi_1"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_2 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_2"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_min_2 = DecimalParameter(
        30.0,
        40.0,
        default=buy_params["buy_rsi_1h_min_2"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_max_2 = DecimalParameter(
        70.0,
        95.0,
        default=buy_params["buy_rsi_1h_max_2"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_diff_2 = DecimalParameter(
        30.0,
        50.0,
        default=buy_params["buy_rsi_1h_diff_2"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_mfi_2 = DecimalParameter(
        30.0,
        56.0,
        default=buy_params["buy_mfi_2"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_bb_offset_2 = DecimalParameter(
        0.97,
        0.999,
        default=buy_params["buy_bb_offset_2"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_bb40_bbdelta_close_3 = DecimalParameter(
        0.005,
        0.06,
        default=buy_params["buy_bb40_bbdelta_close_3"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_bb40_closedelta_close_3 = DecimalParameter(
        0.01,
        0.03,
        default=buy_params["buy_bb40_closedelta_close_3"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_bb40_tail_bbdelta_3 = DecimalParameter(
        0.15,
        0.45,
        default=buy_params["buy_bb40_tail_bbdelta_3"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_ema_rel_3 = DecimalParameter(
        0.97,
        0.999,
        default=buy_params["buy_ema_rel_3"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_bb20_close_bblowerband_4 = DecimalParameter(
        0.96,
        0.99,
        default=buy_params["buy_bb20_close_bblowerband_4"],
        space="buy",
        optimize=True,
        load=True,
    )
    buy_bb20_volume_4 = DecimalParameter(
        1.0,
        20.0,
        default=buy_params["buy_bb20_volume_4"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )

    buy_ema_open_mult_5 = DecimalParameter(
        0.016,
        0.03,
        default=buy_params["buy_ema_open_mult_5"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_bb_offset_5 = DecimalParameter(
        0.98,
        1.0,
        default=buy_params["buy_bb_offset_5"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_ema_rel_5 = DecimalParameter(
        0.97,
        0.999,
        default=buy_params["buy_ema_rel_5"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_ema_open_mult_6 = DecimalParameter(
        0.02,
        0.03,
        default=buy_params["buy_ema_open_mult_6"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_bb_offset_6 = DecimalParameter(
        0.98,
        0.999,
        default=buy_params["buy_bb_offset_6"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_volume_7 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_7"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ema_open_mult_7 = DecimalParameter(
        0.02,
        0.04,
        default=buy_params["buy_ema_open_mult_7"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_7 = DecimalParameter(
        24.0,
        50.0,
        default=buy_params["buy_rsi_7"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ema_rel_7 = DecimalParameter(
        0.97,
        0.999,
        default=buy_params["buy_ema_rel_7"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_volume_8 = DecimalParameter(
        1.0,
        6.0,
        default=buy_params["buy_volume_8"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_8 = DecimalParameter(
        36.0,
        40.0,
        default=buy_params["buy_rsi_8"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_tail_diff_8 = DecimalParameter(
        3.0,
        10.0,
        default=buy_params["buy_tail_diff_8"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_9 = DecimalParameter(
        1.0,
        4.0,
        default=buy_params["buy_volume_9"],
        space="buy",
        decimals=2,
        optimize=True,
        load=True,
    )
    buy_ma_offset_9 = DecimalParameter(
        0.94,
        0.99,
        default=buy_params["buy_ma_offset_9"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_bb_offset_9 = DecimalParameter(
        0.97,
        0.99,
        default=buy_params["buy_bb_offset_9"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_min_9 = DecimalParameter(
        26.0,
        40.0,
        default=buy_params["buy_rsi_1h_min_9"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_max_9 = DecimalParameter(
        70.0,
        90.0,
        default=buy_params["buy_rsi_1h_max_9"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_mfi_9 = DecimalParameter(
        36.0,
        65.0,
        default=buy_params["buy_mfi_9"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_10 = DecimalParameter(
        1.0,
        8.0,
        default=buy_params["buy_volume_10"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ma_offset_10 = DecimalParameter(
        0.93,
        0.97,
        default=buy_params["buy_ma_offset_10"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_bb_offset_10 = DecimalParameter(
        0.97,
        0.99,
        default=buy_params["buy_bb_offset_10"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_10 = DecimalParameter(
        20.0,
        40.0,
        default=buy_params["buy_rsi_1h_10"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_ma_offset_11 = DecimalParameter(
        0.93,
        0.99,
        default=buy_params["buy_ma_offset_11"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_min_inc_11 = DecimalParameter(
        0.005,
        0.05,
        default=buy_params["buy_min_inc_11"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_min_11 = DecimalParameter(
        40.0,
        60.0,
        default=buy_params["buy_rsi_1h_min_11"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_max_11 = DecimalParameter(
        70.0,
        90.0,
        default=buy_params["buy_rsi_1h_max_11"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_11 = DecimalParameter(
        30.0,
        48.0,
        default=buy_params["buy_rsi_11"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_mfi_11 = DecimalParameter(
        36.0,
        56.0,
        default=buy_params["buy_mfi_11"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_12 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_12"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ma_offset_12 = DecimalParameter(
        0.93,
        0.97,
        default=buy_params["buy_ma_offset_12"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_12 = DecimalParameter(
        26.0,
        40.0,
        default=buy_params["buy_rsi_12"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ewo_12 = DecimalParameter(
        2.0,
        6.0,
        default=buy_params["buy_ewo_12"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_13 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_13"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ma_offset_13 = DecimalParameter(
        0.93,
        0.98,
        default=buy_params["buy_ma_offset_13"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_ewo_13 = DecimalParameter(
        -14.0,
        -7.0,
        default=buy_params["buy_ewo_13"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_14 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_14"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ema_open_mult_14 = DecimalParameter(
        0.01,
        0.03,
        default=buy_params["buy_ema_open_mult_14"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_bb_offset_14 = DecimalParameter(
        0.98,
        1.0,
        default=buy_params["buy_bb_offset_14"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_ma_offset_14 = DecimalParameter(
        0.93,
        0.99,
        default=buy_params["buy_ma_offset_14"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_volume_15 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_15"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ema_open_mult_15 = DecimalParameter(
        0.02,
        0.04,
        default=buy_params["buy_ema_open_mult_15"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_ma_offset_15 = DecimalParameter(
        0.93,
        0.99,
        default=buy_params["buy_ma_offset_15"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_15 = DecimalParameter(
        30.0,
        50.0,
        default=buy_params["buy_rsi_15"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ema_rel_15 = DecimalParameter(
        0.97,
        0.999,
        default=buy_params["buy_ema_rel_15"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_volume_16 = DecimalParameter(
        1.0,
        10.0,
        default=buy_params["buy_volume_16"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ma_offset_16 = DecimalParameter(
        0.93,
        0.97,
        default=buy_params["buy_ma_offset_16"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_rsi_16 = DecimalParameter(
        26.0,
        50.0,
        default=buy_params["buy_rsi_16"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ewo_16 = DecimalParameter(
        4.0,
        8.0,
        default=buy_params["buy_ewo_16"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_17 = DecimalParameter(
        0.5,
        8.0,
        default=buy_params["buy_volume_17"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_ma_offset_17 = DecimalParameter(
        0.93,
        0.98,
        default=buy_params["buy_ma_offset_17"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )
    buy_ewo_17 = DecimalParameter(
        -18.0,
        -10.0,
        default=buy_params["buy_ewo_17"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_18 = DecimalParameter(
        1.0,
        6.0,
        default=buy_params["buy_volume_18"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_18 = DecimalParameter(
        16.0,
        32.0,
        default=buy_params["buy_rsi_18"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_bb_offset_18 = DecimalParameter(
        0.98,
        1.0,
        default=buy_params["buy_bb_offset_18"],
        space="buy",
        decimals=3,
        optimize=True,
        load=True,
    )

    buy_rsi_1h_min_19 = DecimalParameter(
        40.0,
        70.0,
        default=buy_params["buy_rsi_1h_min_19"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_chop_min_19 = DecimalParameter(
        20.0,
        60.0,
        default=buy_params["buy_chop_min_19"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_20 = DecimalParameter(
        0.5,
        6.0,
        default=buy_params["buy_volume_20"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    # buy_ema_rel_20 = DecimalParameter(0.97, 0.999, default=buy_params["buy_ema_rel_20"], space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_20 = DecimalParameter(
        20.0,
        36.0,
        default=buy_params["buy_rsi_20"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_20 = DecimalParameter(
        14.0,
        30.0,
        default=buy_params["buy_rsi_1h_20"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    buy_volume_21 = DecimalParameter(
        0.5,
        6.0,
        default=buy_params["buy_volume_21"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    # buy_ema_rel_21 = DecimalParameter(0.97, 0.999, default=buy_params["buy_ema_rel_21"], space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_21 = DecimalParameter(
        10.0,
        28.0,
        default=buy_params["buy_rsi_21"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )
    buy_rsi_1h_21 = DecimalParameter(
        18.0,
        40.0,
        default=buy_params["buy_rsi_1h_21"],
        space="buy",
        decimals=1,
        optimize=True,
        load=True,
    )

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.inf_1h
        )
        # EMA
        informative_1h["ema_15"] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        informative_1h["sma_200"] = ta.SMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)
        # BB
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(informative_1h), window=20, stds=2
        )
        informative_1h["bb_lowerband"] = bollinger["lower"]
        informative_1h["bb_middleband"] = bollinger["mid"]
        informative_1h["bb_upperband"] = bollinger["upper"]
        # Pump protections
        informative_1h["safe_pump_24"] = (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / informative_1h["close"].rolling(24).min()
            )
            < self.buy_pump_threshold_1.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / self.buy_pump_pull_threshold_1.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(24).min())
        )
        informative_1h["safe_pump_36"] = (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / informative_1h["close"].rolling(36).min()
            )
            < self.buy_pump_threshold_2.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / self.buy_pump_pull_threshold_2.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(36).min())
        )
        informative_1h["safe_pump_48"] = (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / informative_1h["close"].rolling(48).min()
            )
            < self.buy_pump_threshold_3.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / self.buy_pump_pull_threshold_3.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(48).min())
        )

        informative_1h["safe_pump_24_strict"] = (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / informative_1h["close"].rolling(24).min()
            )
            < self.buy_pump_threshold_4.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / self.buy_pump_pull_threshold_4.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(24).min())
        )
        informative_1h["safe_pump_36_strict"] = (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / informative_1h["close"].rolling(36).min()
            )
            < self.buy_pump_threshold_5.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / self.buy_pump_pull_threshold_5.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(36).min())
        )
        informative_1h["safe_pump_48_strict"] = (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / informative_1h["close"].rolling(48).min()
            )
            < self.buy_pump_threshold_6.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / self.buy_pump_pull_threshold_6.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(48).min())
        )

        informative_1h["safe_pump_24_loose"] = (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / informative_1h["close"].rolling(24).min()
            )
            < self.buy_pump_threshold_7.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(24).max()
                    - informative_1h["close"].rolling(24).min()
                )
                / self.buy_pump_pull_threshold_7.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(24).min())
        )
        informative_1h["safe_pump_36_loose"] = (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / informative_1h["close"].rolling(36).min()
            )
            < self.buy_pump_threshold_8.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(36).max()
                    - informative_1h["close"].rolling(36).min()
                )
                / self.buy_pump_pull_threshold_8.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(36).min())
        )
        informative_1h["safe_pump_48_loose"] = (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / informative_1h["close"].rolling(48).min()
            )
            < self.buy_pump_threshold_9.value
        ) | (
            (
                (
                    informative_1h["open"].rolling(48).max()
                    - informative_1h["close"].rolling(48).min()
                )
                / self.buy_pump_pull_threshold_9.value
            )
            > (informative_1h["close"] - informative_1h["close"].rolling(48).min())
        )

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BB 40
        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()
        dataframe["closedelta"] = (
            dataframe["close"] - dataframe["close"].shift()
        ).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()

        # BB 20
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # EMA 200
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe["sma_5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma_30"] = ta.SMA(dataframe, timeperiod=30)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe)

        # EWO
        dataframe["ewo"] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Chopiness
        dataframe["chop"] = qtpylib.chopiness(dataframe, 14)

        # Dip protection
        dataframe["safe_dips"] = (
            (
                ((dataframe["open"] - dataframe["close"]) / dataframe["close"])
                < self.buy_dip_threshold_1.value
            )
            & (
                (
                    (dataframe["open"].rolling(2).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_2.value
            )
            & (
                (
                    (dataframe["open"].rolling(12).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_3.value
            )
            & (
                (
                    (dataframe["open"].rolling(144).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_4.value
            )
        )

        dataframe["safe_dips_strict"] = (
            (
                ((dataframe["open"] - dataframe["close"]) / dataframe["close"])
                < self.buy_dip_threshold_5.value
            )
            & (
                (
                    (dataframe["open"].rolling(2).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_6.value
            )
            & (
                (
                    (dataframe["open"].rolling(12).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_7.value
            )
            & (
                (
                    (dataframe["open"].rolling(144).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_8.value
            )
        )

        dataframe["safe_dips_loose"] = (
            (
                ((dataframe["open"] - dataframe["close"]) / dataframe["close"])
                < self.buy_dip_threshold_9.value
            )
            & (
                (
                    (dataframe["open"].rolling(2).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_10.value
            )
            & (
                (
                    (dataframe["open"].rolling(12).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_11.value
            )
            & (
                (
                    (dataframe["open"].rolling(144).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.buy_dip_threshold_12.value
            )
        )

        # Volume
        dataframe["volume_mean_4"] = dataframe["volume"].rolling(4).mean().shift(1)
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(30).mean()

        # Offset
        for i in self.ma_types:
            dataframe[f"{i}_offset_buy"] = (
                self.ma_map[f"{i}"]["calculate"](
                    dataframe, self.base_nb_candles_buy.value
                )
                * self.ma_map[f"{i}"]["low_offset"]
            )
            dataframe[f"{i}_offset_sell"] = (
                self.ma_map[f"{i}"]["calculate"](
                    dataframe, self.base_nb_candles_sell.value
                )
                * self.ma_map[f"{i}"]["high_offset"]
            )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True
        )

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.buy_condition_1_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(50))
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_1h"])
                & (
                    (
                        (dataframe["close"] - dataframe["open"].rolling(36).min())
                        / dataframe["open"].rolling(36).min()
                    )
                    > self.buy_min_inc_1.value
                )
                & (dataframe["rsi_1h"] > self.buy_rsi_1h_min_1.value)
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_max_1.value)
                & (dataframe["rsi"] < self.buy_rsi_1.value)
                & (dataframe["mfi"] < self.buy_mfi_1.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_2_enable.value
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(50))
                & (dataframe["safe_pump_24_strict_1h"])
                & (
                    dataframe["volume_mean_4"] * self.buy_volume_2.value
                    > dataframe["volume"]
                )
                &
                # (dataframe['rsi_1h'] > self.buy_rsi_1h_min_2.value) &
                # (dataframe['rsi_1h'] < self.buy_rsi_1h_max_2.value) &
                (dataframe["rsi"] < dataframe["rsi_1h"] - self.buy_rsi_1h_diff_2.value)
                & (dataframe["mfi"] < self.buy_mfi_2.value)
                & (
                    dataframe["close"]
                    < (dataframe["bb_lowerband"] * self.buy_bb_offset_2.value)
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_3_enable.value
                & (
                    dataframe["close"]
                    > (dataframe["ema_200_1h"] * self.buy_ema_rel_3.value)
                )
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["ema_100_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_pump_36_strict_1h"])
                & dataframe["lower"].shift().gt(0)
                & dataframe["bbdelta"].gt(
                    dataframe["close"] * self.buy_bb40_bbdelta_close_3.value
                )
                & dataframe["closedelta"].gt(
                    dataframe["close"] * self.buy_bb40_closedelta_close_3.value
                )
                & dataframe["tail"].lt(
                    dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta_3.value
                )
                & dataframe["close"].lt(dataframe["lower"].shift())
                & dataframe["close"].le(dataframe["close"].shift())
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_4_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_1h"])
                & (dataframe["close"] < dataframe["ema_50"])
                & (
                    dataframe["close"]
                    < self.buy_bb20_close_bblowerband_4.value
                    * dataframe["bb_lowerband"]
                )
                & (
                    dataframe["volume"]
                    < (
                        dataframe["volume_mean_30"].shift(1)
                        * self.buy_bb20_volume_4.value
                    )
                )
            )
        )

        conditions.append(
            (
                self.buy_condition_5_enable.value
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (
                    dataframe["close"]
                    > (dataframe["ema_200_1h"] * self.buy_ema_rel_5.value)
                )
                & (dataframe["safe_dips"])
                & (dataframe["safe_pump_36_strict_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_5.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (
                    dataframe["close"]
                    < (dataframe["bb_lowerband"] * self.buy_bb_offset_5.value)
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_6_enable.value
                & (dataframe["ema_100_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_loose"])
                & (dataframe["safe_pump_36_strict_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_6.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (
                    dataframe["close"]
                    < (dataframe["bb_lowerband"] * self.buy_bb_offset_6.value)
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_7_enable.value
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_strict"])
                & (
                    dataframe["volume"].rolling(4).mean() * self.buy_volume_7.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_7.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["rsi"] < self.buy_rsi_7.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_8_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_loose"])
                & (dataframe["safe_pump_24_1h"])
                & (dataframe["rsi"] < self.buy_rsi_8.value)
                & (
                    dataframe["volume"]
                    > (dataframe["volume"].shift(1) * self.buy_volume_8.value)
                )
                & (dataframe["close"] > dataframe["open"])
                & (
                    (dataframe["close"] - dataframe["low"])
                    > (
                        (dataframe["close"] - dataframe["open"])
                        * self.buy_tail_diff_8.value
                    )
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_9_enable.value
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (
                    dataframe["volume_mean_4"] * self.buy_volume_9.value
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["ema_20"] * self.buy_ma_offset_9.value
                )
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"] * self.buy_bb_offset_9.value
                )
                & (dataframe["rsi_1h"] > self.buy_rsi_1h_min_9.value)
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_max_9.value)
                & (dataframe["mfi"] < self.buy_mfi_9.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_10_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(24))
                & (dataframe["safe_dips_loose"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_10.value)
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["sma_30"] * self.buy_ma_offset_10.value
                )
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"] * self.buy_bb_offset_10.value
                )
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_10.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_11_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (dataframe["safe_dips_loose"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (dataframe["safe_pump_36_1h"])
                & (dataframe["safe_pump_48_loose_1h"])
                & (
                    (
                        (dataframe["close"] - dataframe["open"].rolling(36).min())
                        / dataframe["open"].rolling(36).min()
                    )
                    > self.buy_min_inc_11.value
                )
                & (
                    dataframe["close"]
                    < dataframe["sma_30"] * self.buy_ma_offset_11.value
                )
                & (dataframe["rsi_1h"] > self.buy_rsi_1h_min_11.value)
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_max_11.value)
                & (dataframe["rsi"] < self.buy_rsi_11.value)
                & (dataframe["mfi"] < self.buy_mfi_11.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_12_enable.value
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(24))
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_12.value)
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["sma_30"] * self.buy_ma_offset_12.value
                )
                & (dataframe["ewo"] > self.buy_ewo_12.value)
                & (dataframe["rsi"] < self.buy_rsi_12.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_13_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(24))
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (dataframe["safe_pump_36_loose_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_13.value)
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["sma_30"] * self.buy_ma_offset_13.value
                )
                & (dataframe["ewo"] < self.buy_ewo_13.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_14_enable.value
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(30))
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(50))
                & (dataframe["safe_dips_loose"])
                & (dataframe["safe_pump_24_1h"])
                & (
                    dataframe["volume_mean_4"] * self.buy_volume_14.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_14.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (
                    dataframe["close"]
                    < (dataframe["bb_lowerband"] * self.buy_bb_offset_14.value)
                )
                & (
                    dataframe["close"]
                    < dataframe["ema_20"] * self.buy_ma_offset_14.value
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_15_enable.value
                & (
                    dataframe["close"]
                    > dataframe["ema_200_1h"] * self.buy_ema_rel_15.value
                )
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips"])
                & (dataframe["safe_pump_36_strict_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.buy_ema_open_mult_15.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["rsi"] < self.buy_rsi_15.value)
                & (
                    dataframe["close"]
                    < dataframe["ema_20"] * self.buy_ma_offset_15.value
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_16_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_strict_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_16.value)
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["ema_20"] * self.buy_ma_offset_16.value
                )
                & (dataframe["ewo"] > self.buy_ewo_16.value)
                & (dataframe["rsi"] < self.buy_rsi_16.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_17_enable.value
                & (dataframe["safe_dips_strict"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_17.value)
                    > dataframe["volume"]
                )
                & (
                    dataframe["close"]
                    < dataframe["ema_20"] * self.buy_ma_offset_17.value
                )
                & (dataframe["ewo"] < self.buy_ewo_17.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_18_enable.value
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(20))
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(44))
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(36))
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(72))
                & (dataframe["safe_dips"])
                & (dataframe["safe_pump_24_strict_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_18.value)
                    > dataframe["volume"]
                )
                & (dataframe["rsi"] < self.buy_rsi_18.value)
                & (
                    dataframe["close"]
                    < (dataframe["bb_lowerband"] * self.buy_bb_offset_18.value)
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_19_enable.value
                & (dataframe["ema_100_1h"] > dataframe["ema_200_1h"])
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(36))
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips"])
                & (dataframe["safe_pump_24_1h"])
                & (dataframe["close"].shift(1) > dataframe["ema_100_1h"])
                & (dataframe["low"] < dataframe["ema_100_1h"])
                & (dataframe["close"] > dataframe["ema_100_1h"])
                & (dataframe["rsi_1h"] > self.buy_rsi_1h_min_19.value)
                & (dataframe["chop"] < self.buy_chop_min_19.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_20_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips"])
                & (dataframe["safe_pump_24_loose_1h"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_20.value)
                    > dataframe["volume"]
                )
                & (dataframe["rsi"] < self.buy_rsi_20.value)
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_20.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_21_enable.value
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (dataframe["safe_dips_strict"])
                & (
                    (dataframe["volume_mean_4"] * self.buy_volume_21.value)
                    > dataframe["volume"]
                )
                & (dataframe["rsi"] < self.buy_rsi_21.value)
                & (dataframe["rsi_1h"] < self.buy_rsi_1h_21.value)
                & (dataframe["volume"] > 0)
            )
        )

        for i in self.ma_types:
            conditions.append(
                (dataframe["close"] < dataframe[f"{i}_offset_buy"])
                & (
                    (dataframe["ewo"] < self.ewo_low.value)
                    | (dataframe["ewo"] > self.ewo_high.value)
                )
                & (dataframe["volume"] > 0)
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "buy"] = 1

        return dataframe

    # non active exit - only using roi as an exit
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df["close"] * 100
    return smadif
