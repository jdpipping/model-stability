from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from bdb_study.synthetic import run_synthetic_study


ROOT = Path(__file__).resolve().parents[2]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class SyntheticEndToEndTests(unittest.TestCase):
    def test_two_size_study_finalizes_and_resume_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "synthetic"
            receipt = run_synthetic_study(run_dir, repo_root=ROOT)
            self.assertEqual(receipt["cells"], 16)
            self.assertEqual(receipt["summary_groups"], 8)
            metrics = pd.read_csv(run_dir / "final" / "metrics.csv")
            self.assertEqual(len(metrics), 16)
            self.assertEqual(set(metrics["n_train"]), {2, 4})
            before = tree_hashes(run_dir)
            second = run_synthetic_study(run_dir, repo_root=ROOT)
            self.assertEqual(second, receipt)
            self.assertEqual(tree_hashes(run_dir), before)


if __name__ == "__main__":
    unittest.main()
