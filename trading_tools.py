# trading_tools.py
from google.adk.tools import FunctionTool
from datetime import datetime
from state import trading_state

def execute_buy_order(ticker: str, quantity: int, price: float) -> str:
    """Execute a buy order for a stock"""
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
    """Execute a sell order for a stock"""
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

buy_tool = FunctionTool(func=execute_buy_order)
sell_tool = FunctionTool(func=execute_sell_order)
portfolio_tool = FunctionTool(func=get_portfolio_status)
cancel_tool = FunctionTool(func=cancel_all_pending_orders)
stop_loss_tool = FunctionTool(func=set_stop_loss)
