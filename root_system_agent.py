from google.adk.agents import SequentialAgent

from mrg_guardian_agent import ModelReliabilityGuardian
from trading_system_agent import trading_system

root_agent = SequentialAgent(
    name="MRG_System",
    sub_agents=[
        ModelReliabilityGuardian,
        trading_system 
    ]
)
