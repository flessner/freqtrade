__author__ = "PoCk3T"
__copyright__ = "The GNU General Public License v3.0"

from datetime import datetime
from typing import Dict

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class WinAvgProfit(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:

        wins = len(results[results['profit_ratio'] > 0])
        avg_profit = results['profit_ratio'].sum() * 100.0
        win_ratio = wins / trade_count

        return -avg_profit * (win_ratio * 2)
