from pathlib import Path

import pytest

from maas_tokenizer.errors import UnknownModelError
from maas_tokenizer.registry import ModelRegistry


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model", action="append", default=None, metavar="PROFILE")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "model(profile): requires model assets")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    choices = config.getoption("model") or []
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    if "all" in choices:
        if len(choices) != 1:
            raise pytest.UsageError("--model all cannot be combined with another model")
        selected = {profile.profile_id for profile in registry.profiles}
    else:
        try:
            selected = {registry.resolve(choice).profile_id for choice in choices}
        except UnknownModelError as error:
            raise pytest.UsageError(str(error)) from error
    for item in items:
        marker = item.get_closest_marker("model")
        if marker is not None and not (set(marker.args) & selected):
            item.add_marker(pytest.mark.skip(reason="model not selected"))

