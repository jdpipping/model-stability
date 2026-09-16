from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.betty.paths import resolve_record_path


ROOT = Path(__file__).resolve().parents[2]
WRAPPERS = (
    "cpu.sbatch", "mig45.sbatch", "mig90.sbatch", "b200.sbatch",
    "mig45_6cpu.sbatch", "mig90_14cpu.sbatch",
)


class BettyRuntimeKitTests(unittest.TestCase):
    def _run_wrapper(self, wrapper: str, command: list[str]) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            fake_bin = Path(temporary)
            srun = fake_bin / "srun"
            srun.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "print(json.dumps({'arguments': sys.argv[1:], 'environment': "
                "{key: os.environ.get(key) for key in "
                "('ZOO_BETTY_PROJECT', 'ZOO_BETTY_IMAGE', 'NVIDIA_VISIBLE_DEVICES', "
                "'TF_USE_LEGACY_KERAS', 'CUBLAS_WORKSPACE_CONFIG')}}))\n"
            )
            srun.chmod(0o755)
            environment = dict(os.environ)
            environment.pop("NVIDIA_VISIBLE_DEVICES", None)
            environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
            environment.update({
                "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "ZOO_BETTY_PROJECT_HOST": str(ROOT),
                "ZOO_BETTY_PROJECT": "/workspace/model-stability",
                "ZOO_BETTY_IMAGE": "fixture-image",
            })
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/betty" / wrapper), *command],
                env=environment, capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)

    def test_wrappers_forward_exact_command_and_runtime_environment(self):
        command = ["python3", "-m", "bdb_study", "--help", "argument with spaces"]
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                result = self._run_wrapper(wrapper, command)
                self.assertEqual(result["arguments"][-len(command):], command)
                self.assertIn("--container-image=fixture-image", result["arguments"])
                environment = result["environment"]
                self.assertEqual(environment["ZOO_BETTY_PROJECT"], "/workspace/model-stability")
                self.assertEqual(environment["TF_USE_LEGACY_KERAS"], "0")
                self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                if wrapper == "cpu.sbatch":
                    self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "void")

    def test_default_benchmark_uses_current_location(self):
        for wrapper, device in (("cpu.sbatch", "cpu"), ("mig45.sbatch", "gpu")):
            with self.subTest(wrapper=wrapper):
                result = self._run_wrapper(wrapper, [])
                self.assertEqual(result["arguments"][-4:], [
                    "python3", "scripts/betty/benchmark_models.py", "--device", device,
                ])

    def test_benchmark_imports_project_package_without_pythonpath(self):
        benchmark = ROOT / "scripts/betty/benchmark_models.py"
        code = (
            "import importlib.util, runpy; "
            f"runpy.run_path({str(benchmark)!r}, run_name='benchmark_import_check'); "
            "print(importlib.util.find_spec('rushing_study.models').origin)"
        )
        # Isolated mode excludes the working directory and inherited PYTHONPATH.
        # Loading without __main__ exercises imports without fitting a model.
        result = subprocess.run(
            [sys.executable, "-I", "-c", code], cwd=benchmark.parent,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(Path(result.stdout.strip()), ROOT / "rushing_study/models.py")

    def test_record_lookup_preserves_original_before_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            current = project / "scripts/betty/probes/record.json"
            current.parent.mkdir(parents=True)
            current.write_text("relocated")
            recorded = "hpc/betty/probes/record.json"
            self.assertEqual(resolve_record_path(project, recorded), current)
            original = project / recorded
            original.parent.mkdir(parents=True)
            original.write_text("different original bytes")
            self.assertEqual(resolve_record_path(project, recorded), original)

    def test_record_lookup_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            for relative in ("../outside", "/outside", "hpc/betty/../../outside"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    resolve_record_path(project, relative)
            target = project / "scripts/betty/probes/record.json"
            target.parent.mkdir(parents=True)
            target.symlink_to(project / "missing")
            with self.assertRaises(ValueError):
                resolve_record_path(project, "hpc/betty/probes/record.json")
            target.unlink()
            (project / "hpc").symlink_to(project / "missing-directory")
            with self.assertRaises(ValueError):
                resolve_record_path(project, "hpc/betty/probes/record.json")


if __name__ == "__main__":
    unittest.main()
