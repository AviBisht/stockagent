# risk_tools.py
from google.adk.tools import FunctionTool
import random

def assess_strategy_performance(strategy_id: str, market_volatility: str) -> dict:
    """
    Custom Tool used by the Guardian. Simulates assessing the trading strategy's 
    risk profile against current market conditions (volatility).
    """
    risk_level = 0
    
    if "momentum" in strategy_id.lower() and "choppy" in market_volatility.lower():
        risk_level = random.randint(3, 5) # High Risk
    elif "mean_reversion" in strategy_id.lower() and "trending" in market_volatility.lower():
        risk_level = random.randint(3, 5) # High Risk
    else:
        risk_level = random.randint(0, 2) # Low Risk
        
    analysis_confidence = round(1.0 - (risk_level / 5.0), 2)

    return {
        "strategy_risk_score": risk_level, 
        "confidence_in_strategy": analysis_confidence, 
        "critical_finding": "Strategy Mismatch with Market Type." if risk_level > 2 else "Performance Normal."
    }

RiskAssessmentTool = FunctionTool(func=assess_strategy_performance)
