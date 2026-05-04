from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ==========================================
# STRATEGY AGENT
# ==========================================
strategy_agent = LlmAgent(
    name="StrategyAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        retry_options=(retry_config)
    ),
    instruction="""You are a trading strategy agent.
    
    Analyze market conditions and generate trading signals:
    - BUY signals when conditions are favorable
    - SELL signals when positions should be closed
    - HOLD signals when no action is needed
    
    Consider:
    - Technical indicators
    - Risk assessment from MRG
    - Current portfolio positions
    
    Output your recommendation with reasoning.""",
    output_key="strategy_signal"
)
