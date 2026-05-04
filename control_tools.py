# control_tools.py
from google.adk.tools import FunctionTool, ToolContext
from state import trading_state
from trading_tools import cancel_all_pending_orders

def exit_loop_tool(tool_context: ToolContext) -> str:
    """Signal that monitoring should stop"""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return "⏹️ Monitoring stopped - conditions unsafe"

def pause_trading(reason: str) -> str:
    """Called by MRG to pause trading"""
    trading_state.pause_trading(reason)
    cancel_all_pending_orders()
    return f"🛑 Trading system paused by MRG: {reason}"

def resume_trading() -> str:
    """Called to resume trading after MRG gives all-clear"""
    trading_state.resume_trading()
    return "🟢 Trading system resumed"

exit_loop = FunctionTool(func=exit_loop_tool)
pause_trading_tool = FunctionTool(func=pause_trading)
resume_trading_tool = FunctionTool(func=resume_trading)
