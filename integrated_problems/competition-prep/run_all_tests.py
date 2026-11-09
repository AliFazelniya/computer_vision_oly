"Run every module's self-test in its own process. Exit code 1 if any fails."
import subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
files = sorted(p for p in root.rglob("*.py") if p.name not in ("__init__.py", "run_all_tests.py", "template_solution.py"))
failed = 0
for p in files:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=300, cwd=str(root))
    ok = r.returncode == 0
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {p.relative_to(root)}  ({time.time() - t0:.1f}s)")
    if not ok: print(r.stdout[-800:], r.stderr[-1500:])
print(f"\n{len(files) - failed}/{len(files)} modules passed")
sys.exit(1 if failed else 0)
