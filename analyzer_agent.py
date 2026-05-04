from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from risk_tools import RiskAssessmentTool
from audit_tools import InternalMetricLogger
from control_tools import exit_loop, pause_trading_tool, resume_trading_tool

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

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
