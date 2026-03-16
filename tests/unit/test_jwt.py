"""Unit tests for JWT auth (Phase 6)."""

from datetime import timedelta

import pytest

from opsagent.auth.jwt import TokenPayload, create_token, decode_token


SECRET = "test-secret-key-for-unit-tests"


class TestJWT:
    def test_create_and_decode(self) -> None:
        token = create_token(team_id="team-01", tier="starter", secret=SECRET)
        payload = decode_token(token, SECRET)
        assert payload.team_id == "team-01"
        assert payload.tier == "starter"

    def test_expired_token_raises(self) -> None:
        from fastapi import HTTPException
        token = create_token(
            team_id="team-02",
            tier="free",
            secret=SECRET,
            expires_in=timedelta(seconds=-1),  # already expired
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token, SECRET)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_invalid_token_raises(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.valid.jwt", SECRET)
        assert exc_info.value.status_code == 401

    def test_wrong_secret_raises(self) -> None:
        from fastapi import HTTPException
        token = create_token(team_id="team-03", tier="team", secret=SECRET)
        with pytest.raises(HTTPException):
            decode_token(token, "wrong-secret")

    def test_enterprise_tier_preserved(self) -> None:
        token = create_token(team_id="team-04", tier="enterprise", secret=SECRET)
        payload = decode_token(token, SECRET)
        assert payload.tier == "enterprise"
