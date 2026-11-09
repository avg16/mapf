"""Package allowlisted project files, models and evidence without private or OS files."""
import hashlib
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copy2(ROOT/"report/project_report.pdf", OUT/"Project_Report.pdf")
    shutil.copy2(ROOT/"report/contributions.pdf", OUT/"Member_Contributions.pdf")
    shutil.copy2(ROOT/"report/abhivansh_contributions.pdf", OUT/"Abhivansh_Contributions.pdf")
    shutil.copy2(ROOT/"report/shivang_contributions.pdf", OUT/"Shivang_Contributions.pdf")
    files = [ROOT/name for name in ("README.md", "requirements.txt", "CONTRIBUTIONS.md")]
    files += sorted((ROOT/"hybrid_mapf").glob("*.py"))
    files += [ROOT/"demo"/name for name in ("index.html", "app.js", "style.css", "README.md")]
    for name in ("policy.pt", "gate.json", "r_order.json", "results.csv", "summary.csv", "sweep.csv", "runs.jsonl",
                 "train_log.txt", "gate_log.txt", "tune_R_log.txt", "exp_log.txt"):
        p=ROOT/"artifacts"/name
        if p.exists():files.append(p)
    files += [ROOT/"report"/name for name in ("project_report.tex", "contributions.tex", "contribution_pages.tex", "abhivansh_contributions.tex", "shivang_contributions.tex", "build_assets.py")]
    files += sorted(p for p in (ROOT/"report/generated").iterdir() if p.suffix in (".png", ".tex", ".json"))
    files.append(Path(__file__).resolve())
    files.append(OUT/"SUBMISSION_README.md")
    archive=OUT/"Project_Code.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED) as z:
        for p in files:z.write(p,"mapf_project/"+str(p.relative_to(ROOT)))
    manifest=[]
    for p in files:
        manifest.append(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(ROOT)))
    (OUT/"CODE_MANIFEST.sha256").write_text("\n".join(manifest)+"\n")
    outputs=[OUT/name for name in ("Project_Report.pdf","Member_Contributions.pdf","Abhivansh_Contributions.pdf","Shivang_Contributions.pdf","Project_Code.zip","CODE_MANIFEST.sha256","SUBMISSION_README.md")]
    with ZipFile(OUT/"Complete_Submission.zip","w",ZIP_DEFLATED) as z:
        for p in outputs:z.write(p,p.name)
    print(f"Packaged {len(files)} source/evidence files; archives verified by ZIP CRC.")
    for name in ("Project_Code.zip","Complete_Submission.zip"):
        with ZipFile(OUT/name) as z:assert z.testzip() is None
        print(name,(OUT/name).stat().st_size,"bytes")


if __name__ == "__main__":
    main()
