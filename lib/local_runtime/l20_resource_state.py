"""Recognize actual resource handoff under the configured model restore policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resource_handoff_complete(value: Mapping[str, Any]) -> bool:
    if value.get("state") in {"restored", "closed"} and value.get("original_service_restored") is True:
        return True
    return (
        value.get("state") == "closed"
        and value.get("restore_original_models") is False
        and value.get("resource_released") is True
        and value.get("original_service_restored") is False
    )
