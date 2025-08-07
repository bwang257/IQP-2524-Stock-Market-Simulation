"""
This code was directly pulled from QuantConnect. The only change made was the adjustment of the weight factor/position exposure multiplier to 2.5.
"""

# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *
import pandas as pd
import functools
import collections
import operator


class MLP_PortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    '''Provides an implementation of IPortfolioConstructionModel that generates percent targets based on the
    Insight.WEIGHT. The target percent holdings of each Symbol is given by the Insight.WEIGHT from the last
    active Insight for that symbol.
    For insights of direction InsightDirection.UP, long targets are returned and for insights of direction
    InsightDirection.DOWN, short targets are returned.
    If the sum of all the last active Insight per symbol is bigger than 1, it will factor down each target
    percent holdings proportionally so the sum is 1.
    It will ignore Insight that have no Insight.WEIGHT value.'''

    def __init__(self, algorithm, model = None, rebalance = Resolution.DAILY, portfolio_bias = PortfolioBias.LONG_SHORT):
        '''Initialize a new instance of InsightWeightingPortfolioConstructionModel
        Args:
            rebalance: Rebalancing parameter. If it is a timedelta, date rules or Resolution, it will be converted into a function.
                              If None will be ignored.
                              The function returns the next expected rebalance time for a given algorithm UTC DateTime.
                              The function returns null if unknown, in which case the function will be called again in the
                              next loop. Returning current time will trigger rebalance.
            portfolio_bias: Specifies the bias of the portfolio (Short, Long/Short, Long)'''
        super().__init__(rebalance, portfolio_bias)
        self.algorithm = algorithm

    def should_create_target_for_insight(self, insight):
        '''Method that will determine if the portfolio construction model should create a
        target for this insight
        Args:
            insight: The insight to create a target for'''
        # Ignore insights that don't have Weight value
        return insight.weight is not None

    def determine_target_percent(self, activeInsights: List[Insight])-> Dict[Insight, float]:
        '''Will determine the target percent for each insight
        Args:
            activeInsights: The active insights to generate a target for'''

        # 1. Temp solution: Sum up weights from the mulitple insights
        Features = {}
        for insight in activeInsights:
            if insight.symbol not in Features.keys():
                Features[insight.symbol] = insight.weight
            else:
                Features[insight.symbol] = Features[insight.symbol] + insight.weight
        
        # 2. Compute long/ short weight sum to adjust long short ratio
        p_sum = 0
        n_sum = 0
        for symbol, weight in Features.items():
            if weight > 0:
                p_sum += weight
            elif weight < 0:
                n_sum += np.abs(weight)

        # 3. return results
        result = {}
        emitted_symbol = []
        weight_sums = sum([np.abs(weight) for weight in Features.values()])
        weight_factor = 2.5
        if weight_sums > 1:
            weight_factor = 1 / weight_sums

        for insight in activeInsights:
            if insight.weight * Features[insight.symbol] > 0:
                if insight.symbol not in emitted_symbol:
                    emitted_symbol.append(insight.symbol)
                    result[insight] = Features[insight.symbol] * weight_factor
        return result

    def get_value(self, insight):
        '''Method that will determine which member will be used to compute the weights and gets its value
        Args:
            insight: The insight to create a target for
        Returns:
            The value of the selected insight member'''
        return abs(insight.weight)


    # Multi-Alpha: 
    def get_target_insights(self) -> List[Insight]:
        return list(self.algorithm.insights.get_active_insights(self.algorithm.utc_time))

