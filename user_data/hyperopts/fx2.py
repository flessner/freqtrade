from datetime import datetime
from pandas import DataFrame
from freqtrade.optimize.hyperopt import IHyperOptLoss


class FX2(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs
    ) -> float:

        # days count
        days_count = (max_date - min_date).days
        trades_per_day = trade_count / days_count

        # win rate
        wins = len(results[results["profit_abs"] > 0])
        draws = len(results[results["profit_abs"] == 0])
        losses = len(results[results["profit_abs"] < 0])
        total = wins + draws + losses
        win_rate = wins / total

        # total profit
        total_profit = results["profit_abs"].sum()
        average_profit = results["profit_ratio"].mean() if len(results) > 0 else 0.0

        # activity penalty
        low_activity_penalty = trades_per_day if trades_per_day < 0.5 else 1
        high_activity_penalty = 1 / trades_per_day if trades_per_day > 4 else 1

        return (
            min(win_rate, 0.95)
            * total_profit
            * low_activity_penalty
            * high_activity_penalty
            * -1
        )
