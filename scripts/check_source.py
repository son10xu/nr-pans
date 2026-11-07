"""Verify source and notebook code cells compile without needing torch/data."""
from pathlib import Path
import ast, json

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "experiments").rglob("*.py")):
    ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    print("PY OK", path.relative_to(root))
for path in sorted((root / "experiments").rglob("*.ipynb")):
    nb=json.loads(path.read_text(encoding="utf8"))
    assert nb['nbformat'] >= 4
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        src=''.join(cell['source'])
        # Notebook magics are handled by IPython, not standard Python.
        src='\n'.join('# '+line if line.lstrip().startswith(('%','!')) else line for line in src.splitlines())
        compile(src, f'{path}:cell{i}', 'exec')
    print("NB OK", path.relative_to(root))
