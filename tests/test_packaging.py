from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_production_requirements_match_company_runtime() -> None:
    expected = [
        "fastapi==0.108.0",
        "gigatoken==0.10.0",
        "jinja2==3.1.6",
        "pydantic==2.8.2",
        "tiktoken==0.12.0",
        "tokenizers==0.22.2",
        "transformers==5.2.0",
        "uvicorn[standard]==0.29.0",
    ]
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements == expected


def test_repository_excludes_dataset_and_oracle_only_modules() -> None:
    assert not (ROOT / "datasets").exists()
    assert not (ROOT / "tools").exists()
    assert not (ROOT / "src/maas_tokenizer/jsonl.py").exists()
    assert not (ROOT / "src/maas_tokenizer/hashing.py").exists()
    assert not (ROOT / "src/maas_tokenizer/contracts.py").exists()


def test_repository_contains_no_model_weights_or_derived_caches() -> None:
    forbidden_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    assert not [
        path
        for path in (ROOT / "model_assets").rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert not list((ROOT / "model_assets").rglob(".cache"))


def test_editable_install_is_importable_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", "from maas_tokenizer.api import app; print(app.title)"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "MaaS Tokenizer"
