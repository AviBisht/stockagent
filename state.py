# state.py
from datetime import datetime

# Application Constants
APP_NAME = "Stock_agent"
USER_ID = "default"
SESSION = "default"

# --- Internal Audit Log ---
INTERNAL_AUDIT_LOG = {}

# --- Memory Storage  ---
MEMORY_BANK = {
    "risk_scores": {},
    "strategies": {}
}

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

# state manager singleton
trading_state = TradingStateManager()
