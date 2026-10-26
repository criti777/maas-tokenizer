from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_ROOTS = ("src", "vendor", "model_assets")
# The pinned official DeepSeek V4.1 implementation carries its own assertions.
# Preserve that third-party source verbatim; enforce this gate on all other code.
UPSTREAM_ASSERT_EXCEPTIONS = {
    Path("vendor/vllm/extracted/deepseek_v41_encoding.py"),
}


def test_production_python_contains_no_assert_statements() -> None:
    violations: list[str] = []

    for root_name in PRODUCTION_ROOTS:
        for path in sorted((PROJECT_ROOT / root_name).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(PROJECT_ROOT)
            if relative_path in UPSTREAM_ASSERT_EXCEPTIONS:
                continue
            violations.extend(
                f"{relative_path}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
            )

    assert not violations, "production assert statements found:\n" + "\n".join(
        violations
    )
