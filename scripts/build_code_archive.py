"""Build a local, explicitly allowlisted experiment-only archive. Never upload.

Manuscript sources, bibliography, editorial text, PDF, data and Git metadata
are deliberately absent. A future publication still requires authorization.
"""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
ALLOW=[
    "README.md", "pyproject.toml", "requirements-lock.txt", "DATA_SOURCES.md",
    "LICENSE", "LICENSE-DATA.md", "CITATION.cff", ".zenodo.json", ".gitignore",
    "src/abo/__init__.py", "src/abo/allocation.py",
    "scripts/download_data.py", "scripts/run_experiments.py", "scripts/robustness.py",
    "scripts/analyze.py", "scripts/verify_outputs.py", "scripts/build_code_archive.py",
    "tests/test_allocation.py", "tests/test_review_claims.py", "scripts/review_analysis.py",
    "scripts/build_repro_archive.py",
]
FORBIDDEN={".tex",".bib",".bst",".sty",".cls",".pdf",".doc",".docx"}


def build():
    out=ROOT/"dist";out.mkdir(exist_ok=True)
    archive=out/"abo-experiments-only.zip"
    for name in ALLOW:
        p=ROOT/name
        if not p.is_file() or p.suffix.lower() in FORBIDDEN or p.is_symlink():
            raise ValueError(f"Not an allowed regular code/documentation file: {name}")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in ALLOW:z.write(ROOT/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==ALLOW
        assert z.testzip() is None
        assert all(Path(name).suffix.lower() not in FORBIDDEN for name in z.namelist())
        # Inspect content too: no accidentally embedded manuscript or bibliography.
        for name in z.namelist():
            body=z.read(name)
            for marker in [bytes([92])+b"documentclass",bytes([92])+b"begin{document}",b"@"+b"article{"]:
                assert marker not in body, (name,marker)
    manifest={"archive":archive.name,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
              "files":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ALLOW},
              "uploaded":False,"manuscript_sources_included":False}
    (out/"archive-audit.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":build()
