from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task2.reproducibility import canonical_sha256, git_state, sha256_file


class Task2ReproducibilityTest(unittest.TestCase):
    def test_canonical_hash_ignores_dictionary_order(self):
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [2, 3]}),
            canonical_sha256({"b": [2, 3], "a": 1}),
        )

    def test_file_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_text("first")
            first = sha256_file(path)
            path.write_text("second")
            self.assertNotEqual(first, sha256_file(path))

    def test_git_state_ignores_untracked_files_but_detects_tracked_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", root], check=True)
            subprocess.run(["git", "-C", root, "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", root, "config", "user.name", "Test"], check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("clean")
            subprocess.run(["git", "-C", root, "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", root, "commit", "-qm", "init"], check=True)
            (root / "untracked.txt").write_text("local only")
            self.assertEqual(git_state(root)["status_porcelain"], "")
            tracked.write_text("dirty")
            self.assertIn("tracked.txt", git_state(root)["status_porcelain"])


if __name__ == "__main__":
    unittest.main()
