"""No credential may live in a repository file.

The repository will be committed and its artifact released anonymously with the paper, so a token
left in a script would be published. Tokens belong in the environment or in ~/.hf_token.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "related papers", "__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pdf", ".png", ".bin", ".pt", ".safetensors", ".npy"}

# Hugging Face user tokens, GitHub tokens and AWS access keys
PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def repo_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts) or path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def test_no_credentials_in_repository_files():
    leaks = []
    for path in repo_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                leaks.append(f"{path.relative_to(ROOT)}: looks like a {name}")  # never print the value
    assert not leaks, "credentials found:\n" + "\n".join(leaks)


def test_the_scanner_would_catch_a_leak(tmp_path):
    assert PATTERNS["Hugging Face token"].search("export HF_TOKEN=hf_" + "a" * 34)
    assert not PATTERNS["Hugging Face token"].search("export HF_TOKEN=\"$HOME/.hf_token\"")


def test_token_file_and_env_files_are_ignored_by_git():
    ignore = (ROOT / ".gitignore").read_text()
    for entry in (".hf_token", "*.token", ".env"):
        assert entry in ignore.splitlines(), entry
