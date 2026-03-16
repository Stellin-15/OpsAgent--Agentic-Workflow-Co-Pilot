"""
Alert source adapter protocol.

Each integration (AlertManager, Grafana, PagerDuty) implements this protocol
to normalise its webhook payload into a list of NormalisedAlert objects.

The adapter pattern means:
  - The core incident creation logic never knows which tool sent the alert
  - Adding a new alert source = adding one new adapter file, nothing else
  - Makes unit testing trivial (test each adapter in isolation with fixture payloads)
"""

from typing import Protocol, runtime_checkable

from opsagent.schemas.alert import NormalisedAlert


@runtime_checkable
class AlertAdapter(Protocol):
    """Convert a raw webhook body dict into normalised alert objects."""

    def normalise(self, body: dict) -> list[NormalisedAlert]: ...

    def can_handle(self, body: dict) -> bool:
        """Return True if this adapter recognises the given payload format."""
        ...
