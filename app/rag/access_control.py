"""Access control: datasource validation and user-based ACL."""
import logging
import os
import secrets

from settings import settings

logger = logging.getLogger(__name__)

_REGISTRY: "UserRegistry | None" = None

_PRIVILEGED_ROLES = frozenset({"ceo", "cxo"})


def _abac_permitted(ds_abac: dict, user_role: str, authorized_ous: set[str]) -> bool:
    """Returns True if the ABAC policy allows access to this datasource.

    Policy (fornix ou-classification.md):
      access_allowed if classification == 'public'
        OR (classification == 'internal' AND ou IN authorized_ous)
        OR (classification == 'confidential'
            AND ou IN authorized_ous AND role IN {ceo, cxo})

    Backward-compatible: missing abac block or missing classification → True.
    Missing ou with a non-public classification → True (OU check skipped).
    """
    classification = ds_abac.get("classification")
    if not classification:
        return True
    if classification == "public":
        return True
    ou = ds_abac.get("ou")
    if not ou:
        return True  # classification set but no OU defined — skip OU check
    if classification == "internal":
        return ou in authorized_ous
    if classification == "confidential":
        return ou in authorized_ous and user_role in _PRIVILEGED_ROLES
    return False  # unknown classification → deny


def load_access_control() -> dict:
    config = settings.access_control
    if "models" in config:
        logger.warning(
            "Legacy key 'models' found in access_control.yaml — ignored; delete this section."
        )
    return config


def get_valid_datasources() -> set[str]:
    config = load_access_control()
    ds = config.get("datasources", {})
    return set(ds.keys()) if ds else set()


def is_valid_datasource(name: str) -> bool:
    config = load_access_control()
    ds = config.get("datasources")
    if ds is None:
        return False  # No config = all denied (default-deny)
    return name in ds


class UserRegistry:
    def __init__(
        self,
        token_to_user: dict[str, str],
        user_to_datasources: dict[str, list[str]],
        user_to_role: dict[str, str] | None = None,
        datasource_abac: dict[str, dict] | None = None,
    ) -> None:
        self._token_to_user = token_to_user
        self._user_to_datasources = user_to_datasources
        self._user_to_role: dict[str, str] = user_to_role or {}
        self._datasource_abac: dict[str, dict] = datasource_abac or {}

    @classmethod
    def build_from_config(cls, config: dict) -> "UserRegistry":
        if "models" in config:
            logger.warning(
                "Legacy key 'models' found in access_control.yaml — ignored; delete this section."
            )
        users = config.get("users", {})
        datasources_cfg = config.get("datasources", {})
        token_to_user: dict[str, str] = {}
        user_to_datasources: dict[str, list[str]] = {}
        user_to_role: dict[str, str] = {}

        for username, user_cfg in users.items():
            env_var = user_cfg.get("api_key_env", "")
            token = os.environ.get(env_var, "") if env_var else ""
            if not token:
                logger.warning(
                    "UserRegistry: env var '%s' for user '%s' is unset or empty — skipping",
                    env_var,
                    username,
                )
                continue
            token_to_user[token] = username
            datasources = list(dict.fromkeys(user_cfg.get("datasources", [])))
            user_to_datasources[username] = datasources
            user_to_role[username] = user_cfg.get("role", "member")

        datasource_abac: dict[str, dict] = {
            ds_name: ds_cfg.get("abac", {})
            for ds_name, ds_cfg in datasources_cfg.items()
        }

        return cls(token_to_user, user_to_datasources, user_to_role, datasource_abac)

    def get_user_by_api_key(self, token: str) -> str | None:
        for stored_token, username in self._token_to_user.items():
            if secrets.compare_digest(stored_token, token):
                return username
        return None

    def get_permitted_datasources_for_user(self, user: str) -> list[str]:
        ds_list = list(self._user_to_datasources.get(user, []))
        user_role = self._user_to_role.get(user, "member")
        # Derive authorized_ous from all explicitly listed datasources
        authorized_ous: set[str] = set()
        for ds in ds_list:
            ou = self._datasource_abac.get(ds, {}).get("ou")
            if ou:
                authorized_ous.add(ou)
        return [
            ds for ds in ds_list
            if _abac_permitted(self._datasource_abac.get(ds, {}), user_role, authorized_ous)
        ]


def set_registry(registry: UserRegistry) -> None:
    global _REGISTRY
    _REGISTRY = registry


def get_user_by_api_key(token: str) -> str | None:
    if _REGISTRY is None:
        return None
    return _REGISTRY.get_user_by_api_key(token)


def get_permitted_datasources_for_user(user: str) -> list[str]:
    if _REGISTRY is None:
        return []
    return _REGISTRY.get_permitted_datasources_for_user(user)


def validate_access_control(config: dict, registry: UserRegistry) -> None:
    valid_datasources = set((config.get("datasources") or {}).keys())
    users = config.get("users", {})

    # Step 1: Schema ref integrity — each user's datasources must be registered
    for username, user_cfg in users.items():
        for ds in user_cfg.get("datasources", []):
            if ds not in valid_datasources:
                raise RuntimeError(
                    f"User '{username}' references unregistered datasource '{ds}'"
                )

    # Step 2: api_key_env name duplicates (before env resolution)
    seen_env_vars: dict[str, str] = {}
    for username, user_cfg in users.items():
        env_var = user_cfg.get("api_key_env", "")
        if env_var in seen_env_vars:
            raise RuntimeError(
                f"Duplicate api_key_env '{env_var}' for users "
                f"'{seen_env_vars[env_var]}' and '{username}'"
            )
        seen_env_vars[env_var] = username

    # Step 3: Resolve tokens (mirrors build_from_config — skip empty)
    resolved: list[tuple[str, str]] = []  # (username, token)
    for username, user_cfg in users.items():
        env_var = user_cfg.get("api_key_env", "")
        token = os.environ.get(env_var, "") if env_var else ""
        if token:
            resolved.append((username, token))

    # Step 4: Resolved token value duplicates
    seen_tokens: dict[str, str] = {}
    for username, token in resolved:
        for seen_token, seen_user in seen_tokens.items():
            if secrets.compare_digest(seen_token, token):
                raise RuntimeError(
                    f"Users '{seen_user}' and '{username}' share the same API key value"
                )
        seen_tokens[token] = username

    # Step 5: Chat × INGEST_API_KEY cross collision
    ingest_key = os.environ.get("INGEST_API_KEY", "")
    if ingest_key:
        for _, chat_token in resolved:
            if secrets.compare_digest(ingest_key, chat_token):
                raise RuntimeError(
                    "A chat API key matches INGEST_API_KEY — keys must be unique"
                )

    # Step 6: ABAC classification consistency
    ds_config = config.get("datasources", {})
    for username, user_cfg in users.items():
        user_role = user_cfg.get("role", "member")
        if user_role in _PRIVILEGED_ROLES:
            continue
        for ds_name in user_cfg.get("datasources", []):
            abac = ds_config.get(ds_name, {}).get("abac", {})
            if abac.get("classification") == "confidential":
                raise RuntimeError(
                    f"User '{username}' (role='{user_role}') lists confidential"
                    f" datasource '{ds_name}' — only ceo/cxo may access confidential data"
                )
