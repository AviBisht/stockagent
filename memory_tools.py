# memory_tools.py
from google.adk.tools import FunctionTool
from datetime import datetime
from state import MEMORY_BANK

def archive_risk_score(ticker: str, risk_data: dict) -> str:
    """
    Store the structured risk assessment data (score, confidence) 
    into the Memory Bank for audit and learning.
    """
    try:
        MEMORY_BANK["risk_scores"][ticker] = {
            "score": risk_data.get('strategy_risk_score', 'N/A'),
            "confidence": risk_data.get('confidence_in_strategy', 'N/A'),
            "timestamp": str(datetime.now())
        }
        return f"✓ Archived {ticker}: Risk Score={risk_data.get('strategy_risk_score')}, Confidence={risk_data.get('confidence_in_strategy')}"
    except Exception as e:
        return f"✗ Error archiving: {str(e)}"

def retrieve_risk_score(ticker: str) -> str:
    """Retrieve past risk assessment for a specific stock."""
    try:
        if ticker in MEMORY_BANK["risk_scores"]:
            data = MEMORY_BANK["risk_scores"][ticker]
            return f"Risk data for {ticker}: Score={data['score']}, Confidence={data['confidence']}, Saved at={data['timestamp']}"
        else:
            return f"No risk data found for {ticker}"
    except Exception as e:
        return f"✗ Error retrieving: {str(e)}"

def archive_strategy(regime_id: str, strategy_data: dict) -> str:
    """Store optimal strategy for a market regime."""
    try:
        MEMORY_BANK["strategies"][regime_id] = strategy_data
        return f"✓ Archived strategy for regime '{regime_id}'"
    except Exception as e:
        return f"✗ Error archiving strategy: {str(e)}"

def retrieve_strategy(regime_id: str) -> str:
    """Retrieve optimal strategy for a market regime."""
    try:
        if regime_id in MEMORY_BANK["strategies"]:
            data = MEMORY_BANK["strategies"][regime_id]
            return f"Strategy for {regime_id}: {data}"
        else:
            return f"No strategy found for regime '{regime_id}'"
    except Exception as e:
        return f"✗ Error retrieving strategy: {str(e)}"

ArchiveRiskTool = FunctionTool(func=archive_risk_score)
RetrieveRiskTool = FunctionTool(func=retrieve_risk_score)
ArchiveStrategyTool = FunctionTool(func=archive_strategy)
RetrieveStrategyTool = FunctionTool(func=retrieve_strategy)
