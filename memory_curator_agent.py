from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from memory_tools import ArchiveRiskTool, RetrieveRiskTool, ArchiveStrategyTool, RetrieveStrategyTool

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
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
