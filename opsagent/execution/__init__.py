"""Safe execution engine for OpsAgent.

Loads action catalogs (YAML), validates parameters, runs commands in
a subprocess with timeout + output capture, and auto-rolls-back on failure.

Public API
----------
    from opsagent.execution import ActionCatalog, SafeExecutor, ExecutionResult
"""

from opsagent.execution.action_catalog import ActionCatalog, ActionDefinition
from opsagent.execution.safe_executor import ExecutionResult, SafeExecutor

__all__ = [
    "ActionCatalog",
    "ActionDefinition",
    "ExecutionResult",
    "SafeExecutor",
]
