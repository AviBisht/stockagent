from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from trading_tools import buy_tool, sell_tool, portfolio_tool, cancel_tool, stop_loss_tool

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ==========================================
# Trade Executor Agent
# ==========================================
trading_executor = LlmAgent(
    name="TradingExecutor",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        retry_options=(retry_config)
    ),
    instruction="""You are a trading execution agent.
    
    Before executing ANY trade:
    1. Check portfolio status to see if trading is active
    2. If trading is PAUSED, refuse all trade requests and explain why
    3. If trading is ACTIVE, proceed with the trade
    
    Always prioritize safety and risk management.
    Never execute trades when trading is paused.""",
    tools=[buy_tool, sell_tool, portfolio_tool, cancel_tool, stop_loss_tool],
    output_key="trade_result"
)
