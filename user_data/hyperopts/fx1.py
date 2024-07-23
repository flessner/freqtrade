from datetime import datetime
from pandas import DataFrame
from freqtrade.optimize.hyperopt import IHyperOptLoss


class FX1(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs
    ) -> float:

        # win rate
        wins = len(results[results["profit_abs"] > 0])
        draws = len(results[results["profit_abs"] == 0])
        losses = len(results[results["profit_abs"] < 0])
        total = wins + draws + losses
        win_rate = wins / total

        # mean profit
        profit_mean = results["profit_ratio"].mean() if len(results) > 0 else 0.0

        # trades per day
        days_count = (max_date - min_date).days
        trades_per_day = trade_count / days_count

        print(profit_mean)

        return (
            min(win_rate, 0.95) * min(profit_mean, 0.05) * min(trades_per_day, 2) * -1
        )
