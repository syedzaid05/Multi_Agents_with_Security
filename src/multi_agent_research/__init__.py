"""Multi-Agent AI Research System.

A small, production-shaped research assistant built from cooperating agents:

* :mod:`multi_agent_research.agents`      - Wikipedia / web / analysis / critic agents
* :mod:`multi_agent_research.llm`         - routed LLM gateway with retries & fallback
* :mod:`multi_agent_research.guardrails`  - input/output validation & injection defence
* :mod:`multi_agent_research.orchestration` - the pipeline that wires everything together
* :mod:`multi_agent_research.app`         - the Streamlit user interface
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
