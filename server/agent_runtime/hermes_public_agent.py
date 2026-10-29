"""Apply CLI-resolved public agent parameters omitted by Hermes' ACP factory."""

from __future__ import annotations

import copy
import importlib
import os
from typing import Any


def apply_public_agent_parameters(agent: Any, config: dict[str, Any] | None = None) -> None:
    """Complete a freshly constructed ACP agent using Hermes' own resolvers.

    Call before a query or child construction. The isolated profile already
    contains the permitted public parameters; this adapter neither reads the
    global profile nor changes routes, output budgets, sampling or compression.
    """
    configuration = importlib.import_module("hermes_cli.config")
    constants = importlib.import_module("hermes_constants")
    budgets = importlib.import_module("agent.iteration_budget")
    config = configuration.load_config_readonly() if config is None else config
    if not isinstance(config, dict):
        raise ValueError("Hermes agent configuration must be a mapping")
    section = config.get("agent")
    section = section if isinstance(section, dict) else {}
    raw_limit = section.get("max_turns")
    if raw_limit is None:
        raw_limit = config.get("max_turns")
    if raw_limit is None:
        raw_limit = os.environ.get("HERMES_MAX_ITERATIONS")
    limit = configuration.resolve_turn_limit(raw_limit)
    reasoning = constants.resolve_reasoning_config(config, getattr(agent, "model", ""))
    previous = getattr(agent, "iteration_budget", None)
    if previous is not None and previous.used != 0:
        raise RuntimeError("Public Hermes parameters must be applied before the first iteration")
    budget = previous if previous is not None and previous.max_total == limit else budgets.IterationBudget(limit)
    initialization = copy.deepcopy(getattr(agent, "_session_init_model_config", {}))
    initialization.update({"reasoning_config": copy.deepcopy(reasoning), "max_iterations": limit})
    agent.reasoning_config = reasoning
    agent.max_iterations = limit
    agent.iteration_budget = budget
    agent._session_init_model_config = initialization
