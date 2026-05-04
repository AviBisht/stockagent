from google.adk.agents import LoopAgent

from analyzer_agent import RiskAnalyzerTool_in_loop
from memory_curator_agent import MemoryAgent_in_loop

ModelReliabilityGuardian = LoopAgent(
    name="MRG_Guardian",
    sub_agents=[
        RiskAnalyzerTool_in_loop,    #  Check risk
        MemoryAgent_in_loop      #  Archive
    ],
    max_iterations=5  # loop cycles
)
