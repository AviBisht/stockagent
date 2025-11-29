1. Problem Description

The Difficulty of Independent Trading

The absence of a dynamic, adaptable safety layer in autonomous trading systems is the main issue we are attempting to resolve.

In certain, predictable market conditions, algorithmic trading models are very successful. But when market regimes shift (from Trending to Choppy, for example) or the model runs into unforeseen "edge cases," the strategy's performance can deteriorate quickly, resulting in catastrophic losses before a human operator can step in.

Why It Matters:

Risk Mitigation: Guaranteed dependability is essential for financial systems. During periods of model uncertainty, a trading system must strictly minimize risk exposure in addition to maximizing profit.

Adaptive Control: When a complex strategy starts to fail, a static set of rules cannot adjust to the qualitative assessments needed (e.g., "Is the market really 'choppy,' or is this just noise?"). To evaluate the model's confidence in real time, we require an intelligent system.

2. Why Agents?
LLM agents are the ideal solution because the problem requires qualitative judgment, nonlinear decision-making, and dynamic tool orchestration, all of which are challenging for traditional code.

Tool Orchestration

The agent seamlessly handles the complex, conditional logic: "If risk is high, call pause_trading, then call exit_loop, and then log_confidence_score."

Qualitative Judgment

The RiskAnalyzerAgent receives quantitative output (a score of 3, confidence of 0.4) and translates it into a critical action (Pause Trading) based on its defined role and instruction set ("Always prioritize system safety").

Stateful Workflow

Agents use the ADK state machine to remember the context (the market is 'Choppy,' the strategy is 'Mean Reversion') across multiple cycles and tool calls, ensuring continuity.

Dynamic Goal Seeking

The SequentialAgent structure allows the system to transition from the monitoring goal (run the loop) to the execution goal (run the trade) without manual intervention, guided by the internal state.

3. What You Created: Overall Architecture
I created the Model Reliability Guardian (MRG) System, a two-phase, hierarchical, multi-agent architecture.

The entire process is managed by a top-level SequentialAgent that forces the system to complete the safety monitoring phase before proceeding to execution.

Phase 1: Model Reliability Guardian (LoopAgent)

This phase handles safety. The LoopAgent runs for a fixed number of cycles, checking risk in each iteration.

Key Agent: The RiskAnalyzerTool_in_loop (LlmAgent) determines if the system should:

Pause and Stop: If strategy_risk_score >= 3 or confidence_in_strategy <= 0.5.

Resume Trading: If conditions are exceptionally good.

Continue: If risk is moderate.

Phase 2: Trading System (SequentialAgent)

This phase handles conditional execution, running only after the monitoring loop is complete.

StrategyAgent: Generates the trading signal (BUY/SELL/HOLD) and parameters, often prompting the user for missing details.

TradingExecutor: Executes the trade using the low-level execute_buy_order or execute_sell_order tools.

4. Demo: Successful Execution

The system successfully executed Scenario A: Low Risk, Full Cycle Completion, and Trade Execution.

Scenario A Summary:

Start Monitoring: The user initiates the process: "Start monitoring the 'Mean Reversion' strategy in a 'Choppy' market for 5 cycles..."

Safety Confirmed: The LoopAgent runs through 5 cycles. The RiskAnalyzerAgent repeatedly logs acceptable risk and confidence levels (approx. 0.7-0.9), calling the InternalMetricLogger tool for audit purposes.

Transition: The loop finishes successfully, and control passes to the TradingSystem.

Parameter Acquisition: The system initially stalls, correctly asking the user for missing trade parameters (Ticker, Quantity).

Execution: After the user provides the missing details ("...to BUY 100 shares of MSFT..."), the agents:

Call archive_strategy to save the trade plan.

Call execute_buy_order to execute the trade.

Result: The system concludes with a final BUY recommendation and confirmation that all safety and execution steps were completed.

5. The Build

Google Agent Development Kit (ADK)

Provides the core hierarchical architecture (SequentialAgent, LoopAgent) and state management.

LLM

Gemini (e.g., gemini-2.5-flash-lite)

Provides the reasoning, qualitative judgment, and function-calling capabilities for the LLM Agents.

Monitoring Tool

InternalMetricLogger (FunctionTool)

Custom Solution: Replaced the unstable external MCP server. Provides reliable, internal logging for risk confidence scores.

Core Functions

Custom FunctionTools (e.g., assess_strategy_performance, pause_trading, execute_buy_order)

Wraps all specific financial and system control logic into callable functions for the LLM agents.

Data Storage

Python dictionaries (MEMORY_BANK)

Used for simple, in-memory state and strategy archival.

6. If I Had More Time, This is What I'd Do

If development time allowed, the following enhancements would push the MRG system toward a robust production state:

Dedicated MCP Service: I would develop and deploy a dedicated MCP server using gRPC (e.g., in a separate Go or Python microservice). This would provide a reliable, network-accessible data layer for logging and retrieving metrics, replacing the internal dictionary and allowing true multi-user access.

Real-Time Data Integration: The Risk Assessment Tool currently simulates data. I would integrate an external tool that pulls real-time market data (e.g., volatility index, order book depth) to make the risk assessment dynamic and market-driven.

Complex Strategy Archival: The MemoryCuratorAgent would be expanded to handle complex strategy objects (JSON serialization of indicator parameters and stop-loss logic) rather than simple dictionaries, ensuring the system can retrieve and restore a full trading setup.

External Alerting: I would integrate a tool to send a high-priority alert (e.g., to a Slack or PagerDuty channel) upon a forced trading pause, ensuring human oversight is immediately notified when the MRG system takes critical protective action.
