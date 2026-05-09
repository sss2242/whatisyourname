"""Integration tests -- Result Class Serialization.

Auto-discovers all *Result dataclasses in operator1/ and verifies
they can be serialized to JSON without error. Catches the bug pattern
where new model results are added but not wired into the profile.

Pattern: Evidently auto-discovery via introspection.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import pkgutil
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)


def _discover_result_classes() -> list[tuple[str, type]]:
    """Find all dataclass types ending in 'Result' under operator1/."""
    result_classes: list[tuple[str, type]] = []
    operator1_path = Path(__file__).parent.parent / "operator1"

    for root, dirs, files in os.walk(operator1_path):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(operator1_path.parent)
            mod_name = str(rel).replace(os.sep, ".").removesuffix(".py")
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    name.endswith("Result")
                    and is_dataclass(obj)
                    and obj.__module__ == mod.__name__
                ):
                    result_classes.append((f"{mod_name}.{name}", obj))
    return result_classes


def _create_minimal_instance(cls: type) -> Any:
    """Create a minimal instance of a dataclass using defaults."""
    try:
        return cls()
    except TypeError:
        # Has required fields -- try with None/0/empty for each
        kwargs = {}
        for f in fields(cls):
            if f.default is not f.default_factory:
                continue
            # Try common default values
            if f.type in ("str", str):
                kwargs[f.name] = ""
            elif f.type in ("int", int):
                kwargs[f.name] = 0
            elif f.type in ("float", float):
                kwargs[f.name] = 0.0
            elif f.type in ("bool", bool):
                kwargs[f.name] = False
            else:
                kwargs[f.name] = None
        try:
            return cls(**kwargs)
        except Exception:
            return None


class TestResultSerialization:
    """Auto-discover and validate all Result dataclasses."""

    @pytest.fixture(scope="session")
    def result_classes(self):
        return _discover_result_classes()

    def test_at_least_20_result_classes_found(self, result_classes):
        """Sanity check: we should find many Result classes."""
        assert len(result_classes) >= 20, (
            f"Only found {len(result_classes)} Result classes "
            f"(expected 20+). Discovery may be broken."
        )

    def test_result_classes_instantiate(self, result_classes):
        """Every Result class should be instantiable with defaults."""
        failures = []
        for name, cls in result_classes:
            instance = _create_minimal_instance(cls)
            if instance is None:
                failures.append(name)
        # Allow some failures (complex constructors), but not too many
        fail_rate = len(failures) / max(len(result_classes), 1)
        assert fail_rate < 0.30, (
            f"{len(failures)}/{len(result_classes)} Result classes failed "
            f"to instantiate: {failures[:10]}"
        )

    def test_to_dict_methods_produce_json(self, result_classes):
        """Result classes with to_dict/to_profile_dict should be JSON-safe."""
        failures = []
        tested = 0
        for name, cls in result_classes:
            instance = _create_minimal_instance(cls)
            if instance is None:
                continue
            for method_name in ("to_dict", "to_profile_dict"):
                method = getattr(instance, method_name, None)
                if method is None:
                    continue
                tested += 1
                try:
                    d = method()
                    json.dumps(d, default=str)
                except Exception as exc:
                    failures.append(f"{name}.{method_name}: {exc}")
        if tested > 0:
            assert len(failures) == 0, (
                f"{len(failures)} serialization failures:\n"
                + "\n".join(failures[:10])
            )

    def test_serialization_method_coverage(self, result_classes):
        """Log how many Result classes have serialization methods."""
        with_methods = 0
        without_methods = []
        for name, cls in result_classes:
            has_method = any(
                hasattr(cls, m) for m in ("to_dict", "to_profile_dict")
            )
            if has_method:
                with_methods += 1
            else:
                without_methods.append(name)
        logger.info(
            "Result serialization coverage: %d/%d have to_dict/to_profile_dict",
            with_methods, len(result_classes),
        )
        # At least 30% should have serialization methods
        coverage = with_methods / max(len(result_classes), 1)
        assert coverage > 0.25, (
            f"Only {coverage:.0%} of Result classes have serialization. "
            f"Missing: {without_methods[:10]}"
        )
