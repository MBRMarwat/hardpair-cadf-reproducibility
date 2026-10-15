from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = (
    ROOT
    / "experiment_parts"
    / "part_01_ham10000_core_experiment"
    / "notebooks"
    / "HardPairNet_Lite.ipynb"
)


def main() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = "".join(document["cells"][1]["source"])
    setup = setup.replace(
        "from pathlib import Path\nimport copy, json, platform",
        "from pathlib import Path\n"
        "import copy, json, platform, sys\n\n"
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "PART_DIR = PROJECT_ROOT / 'experiment_parts' / 'part_01_ham10000_core_experiment'\n"
        "if not PART_DIR.is_dir():\n"
        "    PART_DIR = Path.cwd().resolve()\n"
        "sys.path.insert(0, str(PART_DIR / 'code'))",
    )
    document["cells"][1]["source"] = setup.splitlines(keepends=True)

    configuration = "".join(document["cells"][2]["source"])
    configuration = configuration.replace(
        "config = load_config('config.yaml')",
        "config = load_config(PART_DIR / 'configuration' / 'config.yaml')",
    )
    document["cells"][2]["source"] = configuration.splitlines(keepends=True)
    NOTEBOOK.write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
