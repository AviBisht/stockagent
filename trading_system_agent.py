from google.adk.agents import SequentialAgent

from strategy_agent import strategy_agent
from trading_executor_agent import trading_executor

# ==========================================
# TRADING SYSTEM
# ==========================================
trading_system = SequentialAgent(
    name="TradingSystem",
    sub_agents=[
        strategy_agent,      # Generating  trading signal
        trading_executor     # Execute the trade
    ]
)
