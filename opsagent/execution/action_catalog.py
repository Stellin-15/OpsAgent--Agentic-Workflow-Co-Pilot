"""Action catalog loader.

Reads one or more YAML files from the ``actions/`` directory and exposes a
registry that the executor and API layer can query.

Usage
-----
    catalog = ActionCatalog(Path("actions"))
    catalog.load()
    action = catalog.get("rolling_restart")
    filled_cmd = action.fill_command({"deployment_name": "api", "namespace": "prod"})
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
import structlog

from opsagent.execution.base import ActionSpec, ParameterSpec, SafetyLevel

log = structlog.get_logger(__name__)

# Dangerous patterns that are never allowed in a command template.
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[rRf]+f?\b"),      # rm -rf / rm -fr
    re.compile(r"\bDROP\s+TABLE\b", re.I),   # SQL DROP TABLE
    re.compile(r"\bDROP\s+DATABASE\b", re.I),
    re.compile(r"\bTRUNCATE\b", re.I),
    re.compile(r"\bformat\s+[A-Za-z]:\b", re.I),  # Windows format C:
    re.compile(r">\s*/dev/sd"),              # overwrite block device
    re.compile(r"\bmkfs\b"),                 # filesystem creation
]


class ActionDefinition:
    """Validated, ready-to-use action with helper methods."""

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def fill_command(self, params: dict[str, str]) -> str:
        """Interpolate {placeholders} with supplied params."""
        return self._interpolate(self.spec.command, params)

    def fill_dry_run_command(self, params: dict[str, str]) -> str | None:
        if self.spec.dry_run_command is None:
            return None
        return self._interpolate(self.spec.dry_run_command, params)

    def fill_rollback_command(self, params: dict[str, str]) -> str | None:
        if self.spec.rollback_command is None:
            return None
        return self._interpolate(self.spec.rollback_command, params)

    def validate_params(self, params: dict[str, str]) -> dict[str, str]:
        """
        Merge user-supplied params with defaults; raise on missing required
        params; return the completed param dict.
        """
        resolved: dict[str, str] = {}
        for p in self.spec.parameters:
            if p.name in params:
                resolved[p.name] = str(params[p.name])
            elif p.default is not None:
                resolved[p.name] = str(p.default)
            elif p.required:
                raise ValueError(
                    f"Action '{self.spec.name}' requires parameter '{p.name}'"
                )
        return resolved

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    @staticmethod
    def _interpolate(template: str, params: dict[str, str]) -> str:
        """Replace {key} tokens; raise on unknown placeholders."""
        used: set[str] = set(re.findall(r"\{(\w+)\}", template))
        missing = used - set(params)
        if missing:
            raise ValueError(f"Command template references unknown params: {missing}")
        result = template
        for key, val in params.items():
            result = result.replace(f"{{{key}}}", val)
        return result

    # Convenience proxies
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def requires_approval(self) -> bool:
        return self.spec.requires_approval

    @property
    def safety_level(self) -> SafetyLevel:
        return self.spec.safety_level


class ActionCatalog:
    """Registry of all loaded actions across all catalog YAML files."""

    def __init__(self, catalog_dir: Path | str) -> None:
        self._catalog_dir = Path(catalog_dir)
        self._actions: dict[str, ActionDefinition] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load every ``*.yaml`` file in the catalog directory."""
        yaml_files = sorted(self._catalog_dir.glob("*.yaml"))
        if not yaml_files:
            log.warning("action_catalog.empty", dir=str(self._catalog_dir))
            return

        for path in yaml_files:
            self._load_file(path)

        log.info(
            "action_catalog.loaded",
            total=len(self._actions),
            files=[f.name for f in yaml_files],
        )

    def _load_file(self, path: Path) -> None:
        with path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        for raw in data.get("actions", []):
            try:
                spec = self._parse_action(raw, catalog_file=path.name)
                self._validate_safety(spec)
                defn = ActionDefinition(spec)
                self._actions[spec.name] = defn
                log.debug(
                    "action_catalog.registered",
                    name=spec.name,
                    file=path.name,
                )
            except Exception as exc:
                log.error(
                    "action_catalog.skip",
                    name=raw.get("name", "<unknown>"),
                    file=path.name,
                    error=str(exc),
                )

    @staticmethod
    def _parse_action(raw: dict[str, Any], catalog_file: str) -> ActionSpec:
        params_raw = raw.get("parameters") or []
        parameters: list[ParameterSpec] = []
        for p in params_raw:
            if isinstance(p, dict):
                parameters.append(
                    ParameterSpec(
                        name=p["name"],
                        required=p.get("required", True),
                        default=str(p["default"]) if "default" in p else None,
                        description=p.get("description", ""),
                    )
                )

        return ActionSpec(
            name=raw["name"],
            description=raw.get("description", ""),
            labels=raw.get("labels") or [],
            parameters=parameters,
            command=raw["command"],
            dry_run_command=raw.get("dry_run_command"),
            rollback_command=raw.get("rollback_command"),
            timeout_seconds=int(raw.get("timeout_seconds", 60)),
            requires_approval=bool(raw.get("requires_approval", True)),
            safety_level=SafetyLevel(raw.get("safety_level", "medium")),
            catalog_file=catalog_file,
        )

    @staticmethod
    def _validate_safety(spec: ActionSpec) -> None:
        """Reject commands containing dangerous patterns."""
        for pattern in _BLOCKED_PATTERNS:
            for template in [spec.command, spec.dry_run_command, spec.rollback_command]:
                if template and pattern.search(template):
                    raise ValueError(
                        f"Action '{spec.name}' command contains blocked pattern "
                        f"'{pattern.pattern}'"
                    )

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> ActionDefinition:
        if name not in self._actions:
            raise KeyError(f"Unknown action: '{name}'")
        return self._actions[name]

    def list_all(self) -> list[ActionDefinition]:
        return list(self._actions.values())

    def list_by_label(self, label: str) -> list[ActionDefinition]:
        return [a for a in self._actions.values() if label in a.spec.labels]

    def __len__(self) -> int:
        return len(self._actions)
