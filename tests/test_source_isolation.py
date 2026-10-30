"""Static guard: investigator-side packages must never touch ground-truth labels or scenario parameters."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
GUARDED_PACKAGES = ["controller", "investigation", "analytics"]

# Packages that own ground truth / scenario parameters; guarded code must not import them.
FORBIDDEN_IMPORT_ROOTS = {"investigator_data", "generate_data"}

# Files allowed to *mention* ground-truth terms as strings: they define the leakage
# blocklist / instruct the model not to use ground truth. They must still not import or open it.
MENTION_ALLOWLIST = {
    Path("investigation/state.py"),
    Path("controller/prompts.py"),
}
# The runtime guard must name the term in order to block it; it may build Paths but never read files.
GUARD_FILES = {Path("analytics/path_guard.py")}
FORBIDDEN_STRINGS = ["ground_truth", "scenario_ground_truth", "ground-truth"]


def _guarded_files():
    for pkg in GUARDED_PACKAGES:
        yield from (SRC / pkg).rglob("*.py")


def test_guarded_packages_exist():
    files = list(_guarded_files())
    assert len(files) > 10


def test_guarded_code_does_not_import_generator_or_scenarios():
    offenders = []
    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS:
                    offenders.append(f"{path.relative_to(SRC)}: imports {name}")
    assert not offenders, offenders


def test_guarded_code_does_not_reference_ground_truth_paths():
    offenders = []
    for path in _guarded_files():
        rel = path.relative_to(SRC)
        if rel in MENTION_ALLOWLIST or rel in GUARD_FILES:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_STRINGS:
            if term in text:
                offenders.append(f"{rel}: mentions '{term}'")
    assert not offenders, offenders


def test_allowlisted_files_only_mention_ground_truth_as_inert_strings():
    """Allowlisted files may name the terms but must not build filesystem paths from them."""
    for rel in MENTION_ALLOWLIST:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                assert fn not in {"open", "Path", "os.path.join", "pd.read_csv", "json.load"}, (
                    f"{rel}: unexpected file access call {fn}"
                )


def test_guard_files_never_read_files():
    for rel in GUARD_FILES:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                assert fn not in {"open", "pd.read_csv", "json.load"} and not fn.endswith((".read_text", ".open", ".read_bytes")), (
                    f"{rel}: unexpected file read {fn}"
                )
