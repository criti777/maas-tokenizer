from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_ROOTS = ("src", "vendor", "model_assets")


def test_production_python_contains_no_assert_statements() -> None:
    violations: list[str] = []

    for root_name in PRODUCTION_ROOTS:
        for path in sorted((PROJECT_ROOT / root_name).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(PROJECT_ROOT)
            violations.extend(
                f"{relative_path}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
            )

    assert not violations, "production assert statements found:\n" + "\n".join(
        violations
    )
