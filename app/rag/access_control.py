"""Access control: datasource validation and model-to-datasource mapping."""
from settings import settings


def load_access_control() -> dict:
    return settings.access_control


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


def get_permitted_datasources(model_name: str) -> list[str]:
    config = load_access_control()
    models = config.get("models", {})
    model = models.get(model_name, {})
    return model.get("datasources", [])
