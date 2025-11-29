import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error "
    )
#----main agent-----
from google.adk.agents import (Agent,LlmAgent,SequentialAgent,ParallelAgent)
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.agents import LoopAgent
#=====running the agent and also connecting them====
from google.adk.runners import Runner
import asyncio
import random
from datetime import datetime
#===tools(buitl-in,custom, openAPI)===
from google.adk.tools import (google_search,FunctionTool,AgentTool,ToolContext)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.apps.app import App, ResumabilityConfig
#===managing memory and session===
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.adk.sessions import DatabaseSessionService
from google import adk



retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

mrg_app = App(
    name="MRG_System",
    root_agent=root_agent, 
    resumability_config=ResumabilityConfig(is_resumable=True),
)


runner = Runner(
    agent=root_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

print("✅ Runner instance defined.")

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)
# --- Internal Audit Log ---
INTERNAL_AUDIT_LOG = {}

def log_confidence_score(metric_name: str, value: float, timestamp: str) -> str:
    """
    Internal tool to log key metrics (like confidence) for audit purposes, 
    replacing the external MCP server dependency.
    """
    global INTERNAL_AUDIT_LOG
    if metric_name not in INTERNAL_AUDIT_LOG:
        INTERNAL_AUDIT_LOG[metric_name] = []
        
    log_entry = {"value": value, "timestamp": timestamp}
    INTERNAL_AUDIT_LOG[metric_name].append(log_entry)
    
    print(f"✅ [INTERNAL LOG] {metric_name} recorded: {value}")
    return f"Metric '{metric_name}' logged internally."

# Create the new tool
InternalMetricLogger = FunctionTool(func=log_confidence_score)
print("✅ Internal Metric Logger defined (Replacing MCP).")

print("✅ MCP Tool created")
async def run_session(

    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"

):

    """Helper function to run queries in a session and display responses."""

    print(f"\n### Session: {session_id}")



    # Create or retrieve 

    try:

        session = await session_service.create_session(

            app_name=APP_NAME, user_id=USER_ID, session_id=session_id

        )

    except:

        session = await session_service.get_session(

            app_name=APP_NAME, user_id=USER_ID, session_id=session_id

        )



    # Converting single query -> list

    if isinstance(user_queries, str):

        user_queries = [user_queries]



    # Here we process each query

    for query in user_queries:

        print(f"\nUser > {query}")

        query_content = types.Content(role="user", parts=[types.Part(text=query)])



        #  agent response

        async for event in runner_instance.run_async(

            user_id=USER_ID, session_id=session.id, new_message=query_content

        ):

            if event.is_final_response() and event.content and event.content.parts:

                text = event.content.parts[0].text

                if text and text != "None":

                    print(f"Model: > {text}")





print("✅ Helper functions defined.")

# --- Memory Storage  ---
MEMORY_BANK = {
    "risk_scores": {},
    "strategies": {}
}

# --- Memory Function Storage ---
def archive_risk_score(ticker: str, risk_data: dict) -> str:
    """
    Store the structured risk assessment data (score, confidence) 
    into the Memory Bank for audit and learning.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        risk_data: Dict with 'strategy_risk_score' and 'confidence_in_strategy'
    """
    try:
        # store the data
        MEMORY_BANK["risk_scores"][ticker] = {
            "score": risk_data.get('strategy_risk_score', 'N/A'),
            "confidence": risk_data.get('confidence_in_strategy', 'N/A'),
            "timestamp": str(datetime.now())
        }
        return f"✓ Archived {ticker}: Risk Score={risk_data.get('strategy_risk_score')}, Confidence={risk_data.get('confidence_in_strategy')}"
    except Exception as e:
        return f"✗ Error archiving: {str(e)}"

def retrieve_risk_score(ticker: str) -> str:
    """
    Retrieve past risk assessment for a specific stock.
    
    Args:
        ticker: Stock ticker symbol
    """
    try:
        if ticker in MEMORY_BANK["risk_scores"]:
            data = MEMORY_BANK["risk_scores"][ticker]
            return f"Risk data for {ticker}: Score={data['score']}, Confidence={data['confidence']}, Saved at={data['timestamp']}"
        else:
            return f"No risk data found for {ticker}"
    except Exception as e:
        return f"✗ Error retrieving: {str(e)}"

def archive_strategy(regime_id: str, strategy_data: dict) -> str:
    """
    Store optimal strategy for a market regime.
    
    Args:
        regime_id: Market regime identifier (e.g., 'high_growth_low_vol')
        strategy_data: Dict with strategy details
    """
    try:
        MEMORY_BANK["strategies"][regime_id] = strategy_data
        return f"✓ Archived strategy for regime '{regime_id}'"
    except Exception as e:
        return f"✗ Error archiving strategy: {str(e)}"

def retrieve_strategy(regime_id: str) -> str:
    """
    Retrieve optimal strategy for a market regime.
    
    Args:
        regime_id: Market regime identifier
    """
    try:
        if regime_id in MEMORY_BANK["strategies"]:
            data = MEMORY_BANK["strategies"][regime_id]
            return f"Strategy for {regime_id}: {data}"
        else:
            return f"No strategy found for regime '{regime_id}'"
    except Exception as e:
        return f"✗ Error retrieving strategy: {str(e)}"

# --- Create the Tools ---
ArchiveRiskTool = FunctionTool(func=archive_risk_score)
RetrieveRiskTool = FunctionTool(func=retrieve_risk_score)
ArchiveStrategyTool = FunctionTool(func=archive_strategy)
RetrieveStrategyTool = FunctionTool(func=retrieve_strategy)

print("✅ memory logic done.")


def assess_strategy_performance(strategy_id: str, market_volatility: str) -> dict:
    """
    Custom Tool used by the Guardian. Simulates assessing the trading strategy's 
    risk profile against current market conditions (volatility).
    """
    
    # this would calculate Sharpe Ratio, Drawdown
    risk_level = 0
    
    # Simulation logic
    if "momentum" in strategy_id.lower() and "choppy" in market_volatility.lower():
        risk_level = random.randint(3, 5) # High Risk
    elif "mean_reversion" in strategy_id.lower() and "trending" in market_volatility.lower():
        risk_level = random.randint(3, 5) # High Risk
    else:
        risk_level = random.randint(0, 2) # Low Risk
        
    analysis_confidence = round(1.0 - (risk_level / 5.0), 2)

    return {
        "strategy_risk_score": risk_level, # The metric  report (0=Good, 5=Bad)
        "confidence_in_strategy": analysis_confidence, # The metric for the Watchdog 
        "critical_finding": "Strategy Mismatch with Market Type." if risk_level > 2 else "Performance Normal."
    }

RiskAssessmentTool = FunctionTool(func=assess_strategy_performance)
print("✅ Strategy Risk Assessment Tool Recalibrated.")
APP_NAME = "Stock_agent"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session


def exit_loop_tool(tool_context: ToolContext) -> str:
    """Signal that monitoring should stop"""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return "⏹️ Monitoring stopped - conditions unsafe"

exit_loop = FunctionTool(func=exit_loop_tool)

# when MRG stops the loop this function will be called
def pause_trading(reason: str) -> str:
    """Called by MRG to pause trading"""
    trading_state.pause_trading(reason)
    cancel_all_pending_orders()
    return f"🛑 Trading system paused by MRG: {reason}"

# trading resumed
def resume_trading() -> str:
    """Called to resume trading after MRG gives all-clear"""
    trading_state.resume_trading()
    return "🟢 Trading system resumed"

pause_trading_tool = FunctionTool(func=pause_trading)
resume_trading_tool = FunctionTool(func=resume_trading)

# 3.Risk Analyzer Agent inside the App
RiskAnalyzerTool_in_loop = LlmAgent( 
    name="AnalyzerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=(retry_config)),
          instruction=(
        "You are a risk assessment agent. Your job is to enforce system safety by analyzing trading strategy risk.\n"
        "1. **Analyze Risk:** Use the 'assess_strategy_performance' tool to calculate risk and confidence score.\n"
        "2. **Log Metric:** Use the 'log_confidence_score' tool to publish the resulting confidence score.\n"
        "3. **CRITICAL DECISION LOGIC:**\n"
        "   - **PAUSE/STOP (Risk Detected):** If 'strategy_risk_score' >= 3 OR 'confidence_in_strategy' < 0.5:\n"
        "     * Call 'pause_trading' with a clear reason.\n"
        "     * Call 'exit_loop' to stop the monitoring cycle immediately.\n"
        "   - **RESUME (All Clear):** If 'strategy_risk_score' < 2 AND 'confidence_in_strategy' >= 0.7:\n"
        "     * Call 'resume_trading' to ensure the Trading System is active.\n"
        "   - **CONTINUE:** Otherwise, report the status and allow the monitoring loop to proceed.\n"
        "Always prioritize system safety and provide clear reasoning for decisions."
    ),
    tools=[RiskAssessmentTool, InternalMetricLogger, exit_loop, pause_trading_tool, resume_trading_tool], 
    output_key="risk_assessment"
)

#Memory agent
MemoryAgent_in_loop = LlmAgent( 
    name="MemoryCuratorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=(retry_config)),
        instruction=(
        "You are the financial knowledge archivist. You receive structured data, including risk scores "
        "and market conditions. Your job is to use the memory_tool to archive these metrics for audit "
        "and to retrieve optimal strategy details when requested by the Strategist."
        ),
      tools=[
        ArchiveRiskTool, 
        RetrieveRiskTool,
        ArchiveStrategyTool,
        RetrieveStrategyTool ],
        output_key="archive_status"
       )
   
print("✅ Memory Curator Agent defined and linked to the Memory Bank.")

    
ModelReliabilityGuardian = LoopAgent(
    name="MRG_Guardian",
    sub_agents=[
        RiskAnalyzerTool_in_loop,    #  Check risk
        MemoryAgent_in_loop      #  Archive
    ],
    max_iterations=5  # loop cycles
)
print("✅ MRG Monitoring Sequence defined.")


# ==========================================
# TRADING STATE MANAGER
# ==========================================
class TradingStateManager:
    """Manages trading state across the system"""
    def __init__(self):
        self.trading_active = True
        self.current_positions = {}
        self.pending_orders = []
        self.pause_reason = None
        
    def pause_trading(self, reason: str):
        self.trading_active = False
        self.pause_reason = reason
        print(f"🛑 Trading PAUSED: {reason}")
        
    def resume_trading(self):
        self.trading_active = True
        self.pause_reason = None
        print(f"🟢 Trading RESUMED")
        
    def is_trading_allowed(self) -> bool:
        return self.trading_active

# state manager
trading_state = TradingStateManager()

# ==========================================
# TRADING TOOLS
# ==========================================

def execute_buy_order(ticker: str, quantity: int, price: float) -> str:
    """
    Execute a buy order for a stock
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        quantity: Number of shares to buy
        price: Target price per share
    """
    if not trading_state.is_trading_allowed():
        return f"❌ Order REJECTED: Trading is paused ({trading_state.pause_reason})"
    
    try:
      
        order_id = f"BUY_{ticker}_{datetime.now().timestamp()}"
        trading_state.current_positions[ticker] = {
            "quantity": quantity,
            "entry_price": price,
            "timestamp": datetime.now()
        }
        
        return f"✅ BUY Order Executed: {quantity} shares of {ticker} @ ${price} | Order ID: {order_id}"
    except Exception as e:
        return f"❌ Order FAILED: {str(e)}"

def execute_sell_order(ticker: str, quantity: int, price: float) -> str:
    """
    Execute a sell order for a stock
    
    Args:
        ticker: Stock symbol
        quantity: Number of shares to sell
        price: Target price per share
    """
    if not trading_state.is_trading_allowed():
        return f"❌ Order REJECTED: Trading is paused ({trading_state.pause_reason})"
    
    try:
        order_id = f"SELL_{ticker}_{datetime.now().timestamp()}"
        
        if ticker in trading_state.current_positions:
            del trading_state.current_positions[ticker]
        
        return f"✅ SELL Order Executed: {quantity} shares of {ticker} @ ${price} | Order ID: {order_id}"
    except Exception as e:
        return f"❌ Order FAILED: {str(e)}"

def get_portfolio_status() -> str:
    """Get current portfolio and trading status"""
    status = f"Trading Status: {'🟢 ACTIVE' if trading_state.trading_active else '🛑 PAUSED'}\n"
    
    if not trading_state.trading_active:
        status += f"Pause Reason: {trading_state.pause_reason}\n"
    
    status += f"\nCurrent Positions: {len(trading_state.current_positions)}\n"
    for ticker, position in trading_state.current_positions.items():
        status += f"  • {ticker}: {position['quantity']} shares @ ${position['entry_price']}\n"
    
    return status

def cancel_all_pending_orders() -> str:
    """Emergency function to cancel all pending orders"""
    count = len(trading_state.pending_orders)
    trading_state.pending_orders.clear()
    return f"🚫 Cancelled {count} pending orders"

def set_stop_loss(ticker: str, stop_price: float) -> str:
    """Set a stop-loss order"""
    if ticker not in trading_state.current_positions:
        return f"❌ No position found for {ticker}"
    
    return f"✅ Stop-loss set for {ticker} at ${stop_price}"

# ==========================================
#  Trading Tools
# ==========================================
buy_tool =  FunctionTool(func=execute_buy_order)

sell_tool =  FunctionTool(func=execute_sell_order)

portfolio_tool =  FunctionTool(func=get_portfolio_status)

cancel_tool =  FunctionTool(func=cancel_all_pending_orders)

stop_loss_tool =  FunctionTool(func=set_stop_loss)



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

root_agent = SequentialAgent(
    name="MRG_System",
    sub_agents=[
        ModelReliabilityGuardian,
        trading_system 
    ]
)


print("✅ MRG control tools created")
print("completed trading layer")

trade_query = "The optimal trade is to BUY 100 shares of MSFT at the current market price."

await run_session(
    runner_instance=runner, 
    user_queries=trade_query, 
    session_id="SESSION_A"
)

print("✅ ADK components imported successfully.")
