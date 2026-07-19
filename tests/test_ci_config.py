import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_uses_supported_locked_toolchain() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    development_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    lock_path = ROOT / "requirements-lock.txt"

    python_match = re.search(r'python-version:\s*["\'](\d+)\.(\d+)["\']', workflow)
    action_references = re.findall(r"uses:\s*actions/(?:checkout|setup-python)@([^\s#]+)", workflow)

    assert python_match is not None
    assert tuple(map(int, python_match.groups())) >= (3, 12)
    assert len(action_references) == 2
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_references)
    assert "pip-audit==" in development_requirements
    assert "cache-dependency-path: requirements-lock.txt" in workflow
    assert "python -m pip install --require-hashes -r requirements-lock.txt" in workflow
    assert "python -m ruff format --check ." in workflow
    assert "python -m pip_audit -r requirements-lock.txt" in workflow
    assert lock_path.is_file()
    assert "--hash=sha256:" in lock_path.read_text(encoding="utf-8")
