"""Bundle reviewable project files using an explicit allowlist, excluding keys."""
from pathlib import Path
import zipfile
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
files = [root / name for name in (
    "Baseline_vs_Reranking.ipynb", "baseline_vs_reranking.py", "README.md",
    "REPORT_WORKSHEET.md", "requirements.txt", "requirements.lock.txt",
    "REPORT.md", "REPORT.pdf",
    ".env.example", ".gitignore", "Assignment.pdf",
)]
files += sorted((root / "scripts").glob("*.py"))
files += [p for p in sorted((root / "data").glob("*")) if p.suffix in {".json", ".pdf"}]
files += [p for p in sorted((root / "results").glob("*")) if p.suffix in {".json", ".csv", ".png"}]
key = dotenv_values(root / ".env").get("OPENROUTER_API_KEY", "") or ""
for path in files:
    if key and key.encode() in path.read_bytes():
        raise RuntimeError(f"Credential detected in {path.relative_to(root)}; refusing to package.")
target_dir = root / "dist"
target_dir.mkdir(exist_ok=True)
target = target_dir / "rag_assignment_review.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, Path("rag-baseline-vs-reranking") / path.relative_to(root))
print(f"Created {target.relative_to(root)}: {len(files)} files, {target.stat().st_size:,} bytes.")
print(".env, model weights, caches, logs and virtual environment excluded.")
