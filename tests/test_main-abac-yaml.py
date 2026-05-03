"""ABAC schema validation tests for config/access_control.yaml.

Narrow integration test layer: reads the real YAML file via yaml.safe_load.
No server, no Chroma, no HTTP. Designed to run in < 1 second.

These tests validate the post-migration schema (ABAC attributes per datasource,
role per user) aligned with the fornix OU x classification security policy.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "access_control.yaml"

VALID_CLASSIFICATIONS = {"public", "internal", "confidential"}
VALID_OUS = {"nire-personal", "family", "parents"}
VALID_ROLES = {"ceo", "cxo", "junior", "member"}

NON_TEST_DATASOURCES = [
    "parents-docs",
    "family-docs",
    "nire-docs",
    "nire-work",
    "nire-healthcare",
    "nire-finance",
    "nire-english",
    "nire-youtube",
]
TEST_DATASOURCES = ["test-pytest", "test-ds-mgmt"]
ALL_DATASOURCES = NON_TEST_DATASOURCES + TEST_DATASOURCES


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_abac_schema(config: dict) -> None:
    """Validate the ABAC schema of a loaded access_control.yaml dict.

    Raises AssertionError if any schema rule is violated. Used both for the
    real YAML (via the parametrized tests) and for the regression guard.
    """
    assert "datasources" in config, "missing top-level 'datasources' key"
    assert "users" in config, "missing top-level 'users' key"

    datasources = config["datasources"]
    users = config["users"]

    # Each datasource must have abac.classification in the valid set.
    for ds_name, ds in datasources.items():
        assert "abac" in ds, f"datasource {ds_name!r} missing 'abac' block"
        abac = ds["abac"]
        assert "classification" in abac, (
            f"datasource {ds_name!r} missing 'abac.classification'"
        )
        assert abac["classification"] in VALID_CLASSIFICATIONS, (
            f"datasource {ds_name!r} has invalid classification "
            f"{abac['classification']!r}"
        )
        # Confidential datasources must declare an OU.
        if abac["classification"] == "confidential":
            assert "ou" in abac, (
                f"confidential datasource {ds_name!r} must have 'abac.ou'"
            )

    # Each user must have a role in the valid set.
    for user_name, user in users.items():
        assert "role" in user, f"user {user_name!r} missing 'role'"
        assert user["role"] in VALID_ROLES, (
            f"user {user_name!r} has invalid role {user['role']!r}"
        )

    # No dangling references in users[*].datasources.
    ds_keys = set(datasources.keys())
    for user_name, user in users.items():
        for ds_ref in user.get("datasources", []):
            assert ds_ref in ds_keys, (
                f"user {user_name!r} references unknown datasource {ds_ref!r}"
            )


@pytest.fixture(scope="module")
def config() -> dict:
    return _load_config()


# Test 1: Every datasource has a valid classification.
@pytest.mark.parametrize("ds_name", ALL_DATASOURCES)
def test_classification_valid(config: dict, ds_name: str) -> None:
    ds = config["datasources"][ds_name]
    assert "abac" in ds, f"{ds_name} missing abac block"
    assert "classification" in ds["abac"], f"{ds_name} missing classification"
    assert ds["abac"]["classification"] in VALID_CLASSIFICATIONS


# Test 2: Non-test datasources have a valid OU.
@pytest.mark.parametrize("ds_name", NON_TEST_DATASOURCES)
def test_non_test_datasource_has_valid_ou(config: dict, ds_name: str) -> None:
    ds = config["datasources"][ds_name]
    assert "ou" in ds["abac"], f"{ds_name} missing abac.ou"
    assert ds["abac"]["ou"] in VALID_OUS, (
        f"{ds_name} has invalid ou {ds['abac']['ou']!r}"
    )


# Test 3: Test datasources have NO ou key and classification == "public".
@pytest.mark.parametrize("ds_name", TEST_DATASOURCES)
def test_test_datasource_no_ou_and_public(config: dict, ds_name: str) -> None:
    ds = config["datasources"][ds_name]
    assert "ou" not in ds["abac"], (
        f"test datasource {ds_name} must not have abac.ou"
    )
    assert ds["abac"]["classification"] == "public", (
        f"test datasource {ds_name} must have classification: public"
    )


# Test 4: Specific user roles.
def test_user_roles_specific(config: dict) -> None:
    users = config["users"]
    assert users["nire"]["role"] == "ceo"
    for member_name in ("kyoko", "edge", "lute"):
        assert users[member_name]["role"] == "member", (
            f"{member_name} should have role 'member'"
        )


# Test 5: All users have a role in the valid set.
def test_all_user_roles_in_valid_set(config: dict) -> None:
    for user_name, user in config["users"].items():
        assert "role" in user, f"user {user_name} missing 'role'"
        assert user["role"] in VALID_ROLES, (
            f"user {user_name} has invalid role {user['role']!r}"
        )


# Test 6: Datasource keys exactly match the expected set.
def test_datasource_completeness(config: dict) -> None:
    expected = set(ALL_DATASOURCES)
    actual = set(config["datasources"].keys())
    assert actual == expected, (
        f"datasource keys mismatch: missing={expected - actual}, "
        f"extra={actual - expected}"
    )


# Test 7: No dangling datasource references in users.
def test_no_dangling_user_datasource_refs(config: dict) -> None:
    ds_keys = set(config["datasources"].keys())
    for user_name, user in config["users"].items():
        for ds_ref in user.get("datasources", []):
            assert ds_ref in ds_keys, (
                f"user {user_name} references unknown datasource {ds_ref!r}"
            )


# Test 8: Confidential datasources require an OU.
def test_confidential_requires_ou(config: dict) -> None:
    for ds_name, ds in config["datasources"].items():
        abac = ds.get("abac", {})
        if abac.get("classification") == "confidential":
            assert "ou" in abac, (
                f"confidential datasource {ds_name} must have abac.ou"
            )


# Test 9: Regression guard — _validate_abac_schema rejects missing classification.
def test_validate_abac_schema_rejects_missing_classification() -> None:
    bad_config = {
        "datasources": {
            "broken-ds": {
                "description": "intentionally broken — no classification",
                "abac": {
                    "ou": "nire-personal",
                    # classification deliberately missing
                },
            },
        },
        "users": {
            "alice": {
                "api_key_env": "CHAT_API_KEY_ALICE",
                "role": "member",
                "datasources": ["broken-ds"],
            },
        },
    }
    with pytest.raises(AssertionError):
        _validate_abac_schema(bad_config)
