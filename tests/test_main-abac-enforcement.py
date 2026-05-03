"""Tests for ABAC (Attribute-Based Access Control) enforcement in
``app/rag/access_control.py``.

These are pure unit tests — no external services, no FastAPI client. The
``_abac_permitted`` helper, the ``UserRegistry`` ABAC plumbing, and the
``validate_access_control`` step-6 ABAC consistency check are exercised
directly.

Policy under test (fornix ou-classification.md):
    access_allowed if classification == 'public'
        OR (classification == 'internal' AND ou IN authorized_ous)
        OR (classification == 'confidential'
            AND ou IN authorized_ous AND role IN {ceo, cxo})

Backward-compatible: missing abac block / missing classification → True.
Missing ou with non-public classification → True (OU check skipped).
"""
import os
import sys

# Ensure ``app/`` is importable so ``from rag... import ...`` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Per-user chat tokens used by build_from_config tests. Defaults are only used
# when the real env vars are not already set by the test runner.
os.environ.setdefault("CHAT_API_KEY_NIRE", "test-nire-key")
os.environ.setdefault("CHAT_API_KEY_KYOKO", "test-kyoko-key")
os.environ.setdefault("CHAT_API_KEY_EDGE", "test-edge-key")
os.environ.setdefault("CHAT_API_KEY_LUTE", "test-lute-key")
os.environ.setdefault("INGEST_API_KEY", "test-ingest-key")

import pytest

from rag.access_control import (  # noqa: E402  (import after sys.path edit)
    UserRegistry,
    _abac_permitted,
    validate_access_control,
)


# ---------------------------------------------------------------------------
# TestAbacPermitted — direct ``_abac_permitted`` policy checks
# ---------------------------------------------------------------------------


class TestAbacPermitted:
    # Normal cases ---------------------------------------------------------

    def test_public_no_ou_any_role(self):
        """1. public classification → True regardless of role / ous."""
        assert _abac_permitted({"classification": "public"}, "member", set()) is True
        assert _abac_permitted({"classification": "public"}, "ceo", set()) is True
        assert _abac_permitted({"classification": "public"}, "junior", {"ou-z"}) is True

    def test_public_ou_not_in_authorized(self):
        """2. public bypasses OU check entirely."""
        assert (
            _abac_permitted({"classification": "public", "ou": "x"}, "member", set())
            is True
        )
        assert (
            _abac_permitted(
                {"classification": "public", "ou": "x"}, "member", {"y", "z"}
            )
            is True
        )

    def test_internal_ou_in_authorized(self):
        """3. internal + ou ∈ authorized_ous → True (any role)."""
        assert (
            _abac_permitted(
                {"classification": "internal", "ou": "x"}, "member", {"x"}
            )
            is True
        )
        assert (
            _abac_permitted(
                {"classification": "internal", "ou": "x"}, "junior", {"x", "y"}
            )
            is True
        )

    def test_confidential_ceo_ou_in_authorized(self):
        """4. confidential + ou ∈ authorized + role=ceo → True."""
        assert (
            _abac_permitted(
                {"classification": "confidential", "ou": "x"}, "ceo", {"x"}
            )
            is True
        )

    def test_confidential_cxo_ou_in_authorized(self):
        """5. confidential + ou ∈ authorized + role=cxo → True."""
        assert (
            _abac_permitted(
                {"classification": "confidential", "ou": "x"}, "cxo", {"x"}
            )
            is True
        )

    # Error cases ----------------------------------------------------------

    def test_internal_ou_not_in_authorized(self):
        """6. internal + ou ∉ authorized_ous → False."""
        assert (
            _abac_permitted(
                {"classification": "internal", "ou": "x"}, "member", {"y"}
            )
            is False
        )
        assert (
            _abac_permitted(
                {"classification": "internal", "ou": "x"}, "ceo", set()
            )
            is False
        )

    def test_confidential_member_denied(self):
        """7. confidential + member → False even with matching OU."""
        assert (
            _abac_permitted(
                {"classification": "confidential", "ou": "x"}, "member", {"x"}
            )
            is False
        )

    def test_confidential_junior_denied(self):
        """8. confidential + junior → False even with matching OU."""
        assert (
            _abac_permitted(
                {"classification": "confidential", "ou": "x"}, "junior", {"x"}
            )
            is False
        )

    def test_confidential_ceo_ou_not_authorized(self):
        """9. confidential + ceo + ou ∉ authorized → False (OU still required)."""
        assert (
            _abac_permitted(
                {"classification": "confidential", "ou": "x"}, "ceo", {"y"}
            )
            is False
        )

    def test_unknown_classification_denied(self):
        """10. unknown classification (with ou defined) → False.

        Note: the source code's no-ou early-return ("skip OU check") applies
        before the classification dispatch, so an unknown classification
        *without* an ``ou`` returns True. Only unknown classifications that
        carry an ``ou`` are denied — those reach the final ``return False``
        branch.
        """
        assert (
            _abac_permitted({"classification": "secret", "ou": "x"}, "ceo", {"x"})
            is False
        )
        assert (
            _abac_permitted({"classification": "top-secret", "ou": "x"}, "ceo", {"x"})
            is False
        )
        # member also denied for unknown classification
        assert (
            _abac_permitted({"classification": "secret", "ou": "x"}, "member", {"x"})
            is False
        )

    # Edge cases -----------------------------------------------------------

    def test_empty_dict_backward_compat(self):
        """11. {} → True (no abac block defined)."""
        assert _abac_permitted({}, "member", set()) is True
        assert _abac_permitted({}, "ceo", {"x"}) is True

    def test_only_ou_no_classification(self):
        """12. ou but no classification → True."""
        assert _abac_permitted({"ou": "x"}, "member", set()) is True
        assert _abac_permitted({"ou": "x"}, "junior", {"y"}) is True

    def test_internal_no_ou_skip_check(self):
        """13. classification=internal but no ou → True (OU check skipped)."""
        assert (
            _abac_permitted({"classification": "internal"}, "member", set()) is True
        )

    def test_public_no_ou(self):
        """14. classification=public, no ou → True."""
        assert _abac_permitted({"classification": "public"}, "member", set()) is True


# ---------------------------------------------------------------------------
# TestUserRegistryAbac — get_permitted_datasources_for_user
# ---------------------------------------------------------------------------


class TestUserRegistryAbac:
    def test_member_only_internal_all_returned(self):
        """15. member + only-internal datasources matching their OUs → all returned."""
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a", "ds-b"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "ds-a": {"ou": "ou-x", "classification": "internal"},
                "ds-b": {"ou": "ou-x", "classification": "internal"},
            },
        )
        assert reg.get_permitted_datasources_for_user("alice") == ["ds-a", "ds-b"]

    def test_ceo_internal_and_confidential_all_returned(self):
        """16. ceo with internal AND confidential ds, OUs in their list → all returned."""
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a", "ds-b"]},
            user_to_role={"alice": "ceo"},
            datasource_abac={
                "ds-a": {"ou": "ou-x", "classification": "internal"},
                "ds-b": {"ou": "ou-x", "classification": "confidential"},
            },
        )
        assert reg.get_permitted_datasources_for_user("alice") == ["ds-a", "ds-b"]

    def test_public_no_ou_included_for_member(self):
        """17. public datasource with no ou is included for any role."""
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["pub-ds"]},
            user_to_role={"alice": "member"},
            datasource_abac={"pub-ds": {"classification": "public"}},
        )
        assert reg.get_permitted_datasources_for_user("alice") == ["pub-ds"]

    def test_member_confidential_filtered_out(self):
        """18. member with confidential ds in explicit list → confidential filtered.

        Main enforcement case: even when admin lists a confidential datasource
        for a member, runtime ABAC must hide it.
        """
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a", "ds-b"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "ds-a": {"ou": "ou-x", "classification": "internal"},
                "ds-b": {"ou": "ou-x", "classification": "confidential"},
            },
        )
        assert reg.get_permitted_datasources_for_user("alice") == ["ds-a"]

    def test_unknown_user_empty_list(self):
        """19. Unknown user → []."""
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a"]},
            user_to_role={"alice": "member"},
            datasource_abac={"ds-a": {}},
        )
        assert reg.get_permitted_datasources_for_user("ghost") == []

    def test_two_internal_same_ou_both_pass(self):
        """20. Two internal datasources with same OU both in list → both pass.

        Verifies that authorized_ous is correctly derived from the union of
        OUs of the user's explicit datasource list.
        """
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a", "ds-b"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "ds-a": {"ou": "shared-ou", "classification": "internal"},
                "ds-b": {"ou": "shared-ou", "classification": "internal"},
            },
        )
        assert reg.get_permitted_datasources_for_user("alice") == ["ds-a", "ds-b"]

    def test_two_arg_constructor_backward_compat(self):
        """21. Old 2-arg ``UserRegistry({}, {...})`` constructor still works.

        With no role/abac wiring, every datasource in the user's list is
        returned (full backward compatibility).
        """
        reg = UserRegistry({}, {"alice": ["ds-a", "ds-b"]})
        assert reg.get_permitted_datasources_for_user("alice") == ["ds-a", "ds-b"]


# ---------------------------------------------------------------------------
# TestBuildFromConfigAbac — UserRegistry.build_from_config wires ABAC fields
# ---------------------------------------------------------------------------


class TestBuildFromConfigAbac:
    def test_role_ceo_extracted(self):
        """22. role=ceo from user config flows through to internal map."""
        cfg = {
            "datasources": {"ds-a": {}},
            "users": {
                "alice": {
                    "api_key_env": "CHAT_API_KEY_NIRE",
                    "role": "ceo",
                    "datasources": ["ds-a"],
                }
            },
        }
        reg = UserRegistry.build_from_config(cfg)
        # Indirect check: ceo can access confidential ds with matching ou
        # (build a confidential ds and re-check below).
        assert reg._user_to_role["alice"] == "ceo"

    def test_missing_role_defaults_to_member(self):
        """23. Missing role key defaults to 'member'."""
        cfg = {
            "datasources": {"ds-a": {}},
            "users": {
                "alice": {
                    "api_key_env": "CHAT_API_KEY_NIRE",
                    "datasources": ["ds-a"],
                }
            },
        }
        reg = UserRegistry.build_from_config(cfg)
        assert reg._user_to_role["alice"] == "member"

    def test_abac_block_extracted(self):
        """24. Datasource ``abac`` block is extracted into ``datasource_abac``."""
        cfg = {
            "datasources": {
                "ds-a": {
                    "abac": {"ou": "ou-x", "classification": "internal"},
                },
                "ds-b": {
                    "abac": {"ou": "ou-y", "classification": "confidential"},
                },
            },
            "users": {
                "alice": {
                    "api_key_env": "CHAT_API_KEY_NIRE",
                    "datasources": ["ds-a", "ds-b"],
                }
            },
        }
        reg = UserRegistry.build_from_config(cfg)
        assert reg._datasource_abac["ds-a"] == {
            "ou": "ou-x",
            "classification": "internal",
        }
        assert reg._datasource_abac["ds-b"] == {
            "ou": "ou-y",
            "classification": "confidential",
        }

    def test_missing_abac_block_defaults_empty_dict(self):
        """25. Missing abac block → datasource_abac entry is {}."""
        cfg = {
            "datasources": {"ds-a": {}},
            "users": {
                "alice": {
                    "api_key_env": "CHAT_API_KEY_NIRE",
                    "datasources": ["ds-a"],
                }
            },
        }
        reg = UserRegistry.build_from_config(cfg)
        assert reg._datasource_abac["ds-a"] == {}


# ---------------------------------------------------------------------------
# TestValidateAbacStep6 — validate_access_control step 6 (ABAC consistency)
# ---------------------------------------------------------------------------


def _build_step6_cfg(role: str, classification: str) -> dict:
    """Helper: minimal config with one user and one datasource."""
    return {
        "datasources": {
            "secret-ds": {
                "abac": {"ou": "corp", "classification": classification},
            }
        },
        "users": {
            "alice": {
                "api_key_env": "CHAT_API_KEY_NIRE",
                "role": role,
                "datasources": ["secret-ds"],
            }
        },
    }


class TestValidateAbacStep6:
    def test_member_confidential_raises(self):
        """26. member + confidential in explicit list → RuntimeError mentioning
        username, role, and datasource name."""
        cfg = _build_step6_cfg("member", "confidential")
        registry = UserRegistry.build_from_config(cfg)
        with pytest.raises(RuntimeError) as excinfo:
            validate_access_control(cfg, registry)
        msg = str(excinfo.value)
        assert "alice" in msg
        assert "member" in msg
        assert "secret-ds" in msg

    def test_junior_confidential_raises(self):
        """27. junior + confidential → RuntimeError."""
        cfg = _build_step6_cfg("junior", "confidential")
        registry = UserRegistry.build_from_config(cfg)
        with pytest.raises(RuntimeError):
            validate_access_control(cfg, registry)

    def test_ceo_confidential_no_error(self):
        """28. ceo + confidential → no error (privileged role)."""
        cfg = _build_step6_cfg("ceo", "confidential")
        registry = UserRegistry.build_from_config(cfg)
        validate_access_control(cfg, registry)  # should not raise

    def test_cxo_confidential_no_error(self):
        """29. cxo + confidential → no error (privileged role)."""
        cfg = _build_step6_cfg("cxo", "confidential")
        registry = UserRegistry.build_from_config(cfg)
        validate_access_control(cfg, registry)  # should not raise

    def test_member_internal_no_error(self):
        """30. member + internal → no error (only confidential is restricted)."""
        cfg = _build_step6_cfg("member", "internal")
        registry = UserRegistry.build_from_config(cfg)
        validate_access_control(cfg, registry)  # should not raise


# ---------------------------------------------------------------------------
# TestAbacBackwardCompat — pre-ABAC configs continue to work
# ---------------------------------------------------------------------------


class TestAbacBackwardCompat:
    def test_no_abac_blocks_full_list_returned(self):
        """31. Config without any ``abac:`` keys → full list returned for any user."""
        cfg = {
            "datasources": {
                "ds-a": {},
                "ds-b": {},
                "ds-c": {},
            },
            "users": {
                "alice": {
                    "api_key_env": "CHAT_API_KEY_NIRE",
                    "datasources": ["ds-a", "ds-b", "ds-c"],
                },
            },
        }
        reg = UserRegistry.build_from_config(cfg)
        validate_access_control(cfg, reg)
        assert reg.get_permitted_datasources_for_user("alice") == [
            "ds-a",
            "ds-b",
            "ds-c",
        ]

    def test_idempotency_repeated_calls(self):
        """32. get_permitted_datasources_for_user is idempotent."""
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["ds-a", "ds-b"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "ds-a": {"ou": "ou-x", "classification": "internal"},
                "ds-b": {"ou": "ou-x", "classification": "confidential"},
            },
        )
        result1 = reg.get_permitted_datasources_for_user("alice")
        result2 = reg.get_permitted_datasources_for_user("alice")
        assert result1 == result2
        # Second call must not have side-effected internal state.
        assert result1 == ["ds-a"]


# ---------------------------------------------------------------------------
# TestAbacSecurity — OWASP ASVS V4 access-control invariants
# ---------------------------------------------------------------------------


class TestAbacSecurity:
    def test_member_cannot_get_confidential_even_if_listed(self):
        """33. Privilege escalation prevented: member cannot reach confidential
        even when admin mistakenly added it to their explicit datasource list.
        """
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["confidential-ds"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "confidential-ds": {"ou": "corp", "classification": "confidential"},
            },
        )
        permitted = reg.get_permitted_datasources_for_user("alice")
        assert "confidential-ds" not in permitted
        assert permitted == []

    def test_cross_ou_isolation(self):
        """34. Cross-OU isolation: a user authorized for ``family`` cannot
        access an internal datasource whose OU is ``nire-personal``.

        ``authorized_ous`` is derived from the user's *own* explicit datasource
        list; foreign OUs are never granted just because the datasource
        registry contains them.
        """
        reg = UserRegistry(
            token_to_user={},
            user_to_datasources={"alice": ["family-docs"]},
            user_to_role={"alice": "member"},
            datasource_abac={
                "family-docs": {"ou": "family", "classification": "internal"},
                "nire-personal-docs": {
                    "ou": "nire-personal",
                    "classification": "internal",
                },
            },
        )
        permitted = reg.get_permitted_datasources_for_user("alice")
        assert "nire-personal-docs" not in permitted
        assert permitted == ["family-docs"]
        # Sanity: authorized_ous derivation correctness — alice's authorized OU
        # is exactly {family}, so even directly probing the foreign datasource
        # via _abac_permitted with {family} must deny.
        assert (
            _abac_permitted(
                {"ou": "nire-personal", "classification": "internal"},
                "member",
                {"family"},
            )
            is False
        )
