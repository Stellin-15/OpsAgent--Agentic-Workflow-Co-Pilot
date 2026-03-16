"""Unit tests for the action catalog loader and ActionDefinition helpers."""

import textwrap
from pathlib import Path

import pytest
import yaml

from opsagent.execution.action_catalog import ActionCatalog, ActionDefinition
from opsagent.execution.base import SafetyLevel


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    """Create a minimal YAML catalog in a temp directory."""
    content = textwrap.dedent("""\
        actions:
          - name: test_action
            description: "A test action"
            labels: [test, kubernetes]
            parameters:
              - name: deployment_name
                required: true
                description: "The deployment name"
              - name: namespace
                required: false
                default: "default"
                description: "K8s namespace"
            command: "kubectl rollout restart deployment/{deployment_name} -n {namespace}"
            dry_run_command: "kubectl rollout restart deployment/{deployment_name} -n {namespace} --dry-run=client"
            rollback_command: "kubectl rollout undo deployment/{deployment_name} -n {namespace}"
            timeout_seconds: 60
            requires_approval: true
            safety_level: medium

          - name: read_only_action
            description: "A read-only action"
            labels: [test, diagnose]
            parameters: []
            command: "kubectl get pods -n default"
            dry_run_command: null
            rollback_command: null
            timeout_seconds: 30
            requires_approval: false
            safety_level: low
    """)
    (tmp_path / "test.yaml").write_text(content)
    return tmp_path


class TestActionCatalogLoad:
    def test_loads_actions(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        assert len(catalog) == 2

    def test_get_known_action(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")
        assert action.name == "test_action"
        assert action.safety_level == SafetyLevel.MEDIUM

    def test_get_unknown_action_raises(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        with pytest.raises(KeyError):
            catalog.get("nonexistent_action")

    def test_list_by_label(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        diagnose = catalog.list_by_label("diagnose")
        assert len(diagnose) == 1
        assert diagnose[0].name == "read_only_action"

    def test_empty_dir_loads_without_error(self, tmp_path: Path) -> None:
        catalog = ActionCatalog(tmp_path)
        catalog.load()
        assert len(catalog) == 0


class TestActionDefinitionFillCommand:
    def test_fill_command_with_defaults(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")

        params = action.validate_params({"deployment_name": "api"})
        assert params["namespace"] == "default"
        cmd = action.fill_command(params)
        assert cmd == "kubectl rollout restart deployment/api -n default"

    def test_fill_command_overrides_default(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")

        params = action.validate_params({"deployment_name": "api", "namespace": "prod"})
        cmd = action.fill_command(params)
        assert "prod" in cmd

    def test_missing_required_param_raises(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")

        with pytest.raises(ValueError, match="deployment_name"):
            action.validate_params({})

    def test_fill_dry_run_command(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")

        params = action.validate_params({"deployment_name": "api"})
        dry = action.fill_dry_run_command(params)
        assert dry is not None
        assert "--dry-run=client" in dry

    def test_fill_dry_run_none_when_not_defined(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("read_only_action")

        dry = action.fill_dry_run_command({})
        assert dry is None

    def test_fill_rollback_command(self, catalog_dir: Path) -> None:
        catalog = ActionCatalog(catalog_dir)
        catalog.load()
        action = catalog.get("test_action")

        params = action.validate_params({"deployment_name": "api"})
        rollback = action.fill_rollback_command(params)
        assert rollback is not None
        assert "undo" in rollback


class TestSafetyValidation:
    def test_blocked_rm_rf(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            actions:
              - name: dangerous
                description: "Dangerous action"
                labels: []
                parameters: []
                command: "rm -rf /tmp/something"
                dry_run_command: null
                rollback_command: null
                timeout_seconds: 10
                requires_approval: true
                safety_level: critical
        """)
        (tmp_path / "danger.yaml").write_text(content)
        catalog = ActionCatalog(tmp_path)
        catalog.load()
        # The dangerous action should have been skipped
        assert len(catalog) == 0

    def test_blocked_drop_table(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            actions:
              - name: sql_danger
                description: "SQL danger"
                labels: []
                parameters: []
                command: "psql -c 'DROP TABLE incidents'"
                dry_run_command: null
                rollback_command: null
                timeout_seconds: 10
                requires_approval: true
                safety_level: critical
        """)
        (tmp_path / "sql.yaml").write_text(content)
        catalog = ActionCatalog(tmp_path)
        catalog.load()
        assert len(catalog) == 0
