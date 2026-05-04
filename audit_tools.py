# audit_tools.py
from google.adk.tools import FunctionTool
from state import INTERNAL_AUDIT_LOG

def log_confidence_score(metric_name: str, value: float, timestamp: str) -> str:
    """
    Internal tool to log key metrics (like confidence) for audit purposes, 
    replacing the external MCP server dependency.
    """
    if metric_name not in INTERNAL_AUDIT_LOG:
        INTERNAL_AUDIT_LOG[metric_name] = []
        
    log_entry = {"value": value, "timestamp": timestamp}
    INTERNAL_AUDIT_LOG[metric_name].append(log_entry)
    
    print(f"✅ [INTERNAL LOG] {metric_name} recorded: {value}")
    return f"Metric '{metric_name}' logged internally."

InternalMetricLogger = FunctionTool(func=log_confidence_score)
