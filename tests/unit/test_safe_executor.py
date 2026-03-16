"""Unit tests for SafeExecutor (subprocess execution engine)."""

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opsagent.execution.action_catalog import ActionCatalog
from opsagent.execution.base import ExecutionResult
from opsagent.execution.safe_executor import ExecutionRequest, SafeExecutor


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        actions:
          - name: echo_test
            description: "Echo test"
            labels: [test]
            parameters:
              - name: message
                required: true
            command: "echo {message}"
            dry_run_command: "echo dry-{message}"
            rollback_command: null
            timeout_seconds: 5
            requires_approval: false
            safety_level: low

          - name: failing_action
            description: "Always fails"
            labels: [test]
            parameters: []
            command: "exit 1"
            dry_run_command: null
            rollback_command: "echo rollback-executed"
            timeout_seconds: 5
            requires_approval: false
            safety_level: medium
    """)
    (tmp_path / "test.yaml").write_text(content)
    return tmp_path


@pytest.fixture
def catalog(catalog_dir: Path) -> ActionCatalog:
    cat = ActionCatalog(catalog_dir)
    cat.load()
    return cat


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


class TestSafeExecutorRun:
    @pytest.mark.asyncio
    async def test_successful_command(self, catalog, mock_session) -> None:
        executor = SafeExecutor()
        action = catalog.get("echo_test")

        with patch(
            "opsagent.repositories.action_log_repo.ActionLogRepo.log_action",
            new_callable=AsyncMock,
        ):
            request = ExecutionRequest(
                action=action,
                params={"message": "hello"},
                incident_id="01TESTINCIDENT00000001",
                approved_by="test_user",
            )
            result = await executor.run(request, mock_session)

        assert result.exit_code == 0
        assert result.succeeded is True
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_dry_run_uses_dry_run_command(self, catalog, mock_session) -> None:
        executor = SafeExecutor()
        action = catalog.get("echo_test")

        with patch(
            "opsagent.repositories.action_log_repo.ActionLogRepo.log_action",
            new_callable=AsyncMock,
        ):
            request = ExecutionRequest(
                action=action,
                params={"message": "world"},
                incident_id="01TESTINCIDENT00000002",
                approved_by="test_user",
                dry_run=True,
            )
            result = await executor.run(request, mock_session)

        # dry-run command = "echo dry-{message}"
        assert "dry-world" in result.stdout

    @pytest.mark.asyncio
    async def test_nonexistent_binary_returns_127(self, catalog, mock_session) -> None:
        executor = SafeExecutor()
        action = catalog.get("echo_test")

        with patch(
            "opsagent.repositories.action_log_repo.ActionLogRepo.log_action",
            new_callable=AsyncMock,
        ), patch.object(
            action, "fill_command", return_value="nonexistent_binary_xyz arg"
        ):
            request = ExecutionRequest(
                action=action,
                params={"message": "x"},
                incident_id="01TESTINCIDENT00000003",
                approved_by="test_user",
            )
            result = await executor.run(request, mock_session)

        assert result.exit_code == 127
        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_action_log_written(self, catalog, mock_session) -> None:
        executor = SafeExecutor()
        action = catalog.get("echo_test")
        log_calls = []

        async def _fake_log(**kwargs):
            log_calls.append(kwargs)
            return MagicMock()

        with patch(
            "opsagent.repositories.action_log_repo.ActionLogRepo.log_action",
            side_effect=_fake_log,
        ):
            request = ExecutionRequest(
                action=action,
                params={"message": "audit-test"},
                incident_id="01TESTINCIDENT00000004",
                approved_by="test_auditor",
            )
            await executor.run(request, mock_session)

        assert len(log_calls) == 1
        assert log_calls[0]["incident_id"] == "01TESTINCIDENT00000004"
        assert log_calls[0]["approved_by"] == "test_auditor"


class TestExecutionResult:
    def test_succeeded_true_on_zero_exit(self) -> None:
        r = ExecutionResult(
            action_name="x",
            command="echo",
            exit_code=0,
            stdout="",
            stderr="",
        )
        assert r.succeeded is True

    def test_succeeded_false_on_nonzero_exit(self) -> None:
        r = ExecutionResult(
            action_name="x",
            command="exit 1",
            exit_code=1,
            stdout="",
            stderr="",
        )
        assert r.succeeded is False

    def test_succeeded_false_on_timeout(self) -> None:
        r = ExecutionResult(
            action_name="x",
            command="sleep 999",
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
        assert r.succeeded is False
