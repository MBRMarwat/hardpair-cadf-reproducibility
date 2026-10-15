from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


PART_DIR = Path(__file__).resolve().parents[1]
source_path = PART_DIR / "notebooks" / "CADF_Net_v3_Experiments.ipynb"
destination = PART_DIR / "analysis_and_results" / "smoke" / "CADF_Net_v3_Experiments_smoke_executed.ipynb"
notebook = nbformat.read(source_path, as_version=4)
replaced = False
for cell in notebook.cells:
    if cell.cell_type == "code" and "SMOKE = False" in cell.source:
        cell.source = cell.source.replace("SMOKE = False", "SMOKE = True", 1)
        replaced = True
        break
if not replaced:
    raise RuntimeError("Could not locate the SMOKE configuration cell")
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="histo",
    resources={"metadata": {"path": str(PART_DIR)}},
)
client.execute()
destination.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, destination)
print(destination)
