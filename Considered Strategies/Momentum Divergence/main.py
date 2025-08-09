# region imports
from datetime import datetime
from AlgorithmImports import *
from alpha import adapted_alpha
# endregion


class CompetitionAlgorithm(QCAlgorithm):
    
    def Initialize(self):

        # Backtest parameters
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(1000000)
        self.SetWarmUp(timedelta(days=300))

        # Parameters:
        self.final_universe_size = 50

        # Universe selection
        self.rebalanceTime = self.time

        self.add_universe(self.equity_filter)

        self.UniverseSettings.Resolution = Resolution.HOUR

        self.set_portfolio_construction(self.MyPCM())
        self.set_alpha(adapted_alpha(self))
        self.set_execution(VolumeWeightedAveragePriceExecutionModel())
        self.add_risk_management(NullRiskManagementModel())
 
        # set account type
        #self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)


    def equity_filter(self, data):
        self.Log("in filter for equities")
        # Rebalancing monthly
        if self.Time <= self.rebalanceTime:
            return self.Universe.Unchanged
        self.rebalanceTime = self.Time + timedelta(days=30)
        
        sortedByDollarVolume = sorted(data, key=lambda x: x.DollarVolume, reverse=True)
        final = [x.Symbol for x in sortedByDollarVolume if x.HasFundamentalData and x.price > 10 and x.MarketCap > 2000000000][:self.final_universe_size]
        self.Log("coming out of course: " + str(len(final)))
        return final
    class MyPCM(InsightWeightingPortfolioConstructionModel): 
        # override to set leverage higher
        def CreateTargets(self, algorithm, insights): 
            targets = super().CreateTargets(algorithm, insights) 
            return [PortfolioTarget(x.Symbol, x.Quantity * 1.85) for x in targets]
        



        
        



    








            

            

     
