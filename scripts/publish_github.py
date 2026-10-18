"""Publish the validated, secret-scanned bundle to a NEW PUBLIC GitHub repo.

Run from your ordinary terminal if the agent's network is restricted:
    .venv/bin/python scripts/publish_github.py
Requires an authenticated GitHub CLI and a configured Git author identity.
The original folder and any existing Git repositories are left in place.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("name", nargs="?", default="rag-baseline-vs-reranking-gdpr")
parser.add_argument("--prepare-only", action="store_true",
                    help="Build, scan and commit the upload locally without contacting GitHub.")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.name):
    parser.error("Supply a repository name, without an owner or URL.")
gh = shutil.which("gh")
git = shutil.which("git")
if not git or (not args.prepare_only and not gh):
    raise SystemExit("Install Git and GitHub CLI, then run gh auth login before publishing.")

def run(command, cwd=root, capture=False):
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout

run([sys.executable, str(root / "scripts/validate_results.py"), "--require-answers"])
run([sys.executable, str(root / "scripts/build_submission.py")])
repo = root / "dist" / args.name
with zipfile.ZipFile(root / "dist/rag_assignment_review.zip") as bundle:
    if not repo.exists():
        staging = Path(tempfile.mkdtemp(prefix="github-", dir=root / "dist"))
        bundle.extractall(staging)
        shutil.move(str(staging / "rag-baseline-vs-reranking"), repo)
        staging.rmdir()
        run([git, "init", "-b", "main"], cwd=repo)
        # Fail before creating an external repository if Git identity is missing.
        run([git, "var", "GIT_AUTHOR_IDENT"], cwd=repo, capture=True)
        run([git, "add", "."], cwd=repo)
        run([git, "commit", "-m", "Implement reproducible RAG baseline and cross-encoder experiment"], cwd=repo)
    else:
        expected = set()
        for entry in bundle.infolist():
            relative = Path(entry.filename).relative_to("rag-baseline-vs-reranking")
            expected.add(relative.as_posix())
            if not (repo / relative).is_file() or (repo / relative).read_bytes() != bundle.read(entry):
                raise SystemExit(f"Prepared checkout differs from current bundle: {relative}. Use a fresh repository name or review it manually.")
        tracked = set(run([git, "ls-files", "-z"], cwd=repo, capture=True).rstrip("\0").split("\0"))
        if tracked != expected or run([git, "status", "--porcelain"], cwd=repo, capture=True).strip():
            raise SystemExit("Prepared checkout has unexpected files or changes; review it before publishing.")
if args.prepare_only:
    print(f"\nPrepared and committed: {repo}\nNo GitHub repository has been created yet.")
    sys.exit(0)
run([gh, "auth", "status"])
owner = run([gh, "api", "user", "--jq", ".login"], capture=True).strip()
if not re.fullmatch(r"[A-Za-z0-9-]+", owner):
    raise SystemExit("Could not determine the authenticated GitHub account.")
wrapper = shutil.which("gh-axi")
known_wrapper = Path.home() / ".nvm/versions/node/v24.18.0/bin/gh-axi"
if not wrapper and known_wrapper.is_file():
    wrapper = str(known_wrapper)
full_name = f"{owner}/{args.name}"
run([wrapper or gh, "repo", "create", full_name, "--public", "--description",
     "Reproducible RAG experiment: cosine baseline vs cross-encoder re-ranking, with executed notebook and real model answers."], cwd=repo)
url = f"https://github.com/{full_name}"
run([git, "remote", "add", "origin", url + ".git"], cwd=repo)
# Use the authenticated CLI as a credential helper for this push only.
# The fixed helper command passes credentials directly to Git, never to stdout.
run([git, "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
     "push", "-u", "origin", "main"], cwd=repo)
(root / "results/github_repository.json").write_text(json.dumps(
    {"url": url, "visibility": "public", "uploaded": True}, indent=2) + "\n")
print(f"\nPublished: {url}\nLocal Git checkout: {repo}")
