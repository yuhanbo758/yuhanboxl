"""版本递增脚本的单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.bump_version import increment_version, read_version, write_version


class BumpVersionTests(unittest.TestCase):
    """验证进位规则以及版本字段的精确更新行为。"""

    def test_normal_increment(self) -> None:
        self.assertEqual(increment_version("0.6.8"), "0.6.9")

    def test_patch_carry(self) -> None:
        self.assertEqual(increment_version("0.6.9"), "0.7.0")

    def test_major_carry(self) -> None:
        self.assertEqual(increment_version("0.9.9"), "1.0.0")

    def test_invalid_version_fails(self) -> None:
        with self.assertRaises(ValueError):
            increment_version("0.6")

    def test_only_target_field_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "setup.py"
            original = "dependency = 'demo==0.3.6'\nsetup(version='0.3.6')\n"
            version_file.write_text(original, encoding="utf-8")

            self.assertEqual(read_version(version_file), "0.3.6")
            write_version(version_file, "0.3.7")

            updated = version_file.read_text(encoding="utf-8")
            self.assertIn("dependency = 'demo==0.3.6'", updated)
            self.assertIn("setup(version='0.3.7')", updated)


if __name__ == "__main__":
    unittest.main()
