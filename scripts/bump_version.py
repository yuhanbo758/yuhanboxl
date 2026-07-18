"""按每位逢十进一的规则更新 setup.py 中的唯一版本号。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# 只允许匹配 setup() 中形如 version='1.2.3' 的三段式版本字段，
# 从而避免误改依赖版本、文档示例或其他无关数字。
VERSION_PATTERN = re.compile(
    r"(?P<prefix>\bversion\s*=\s*['\"])(?P<version>\d+\.\d+\.\d+)(?P<suffix>['\"])",
    re.MULTILINE,
)


def increment_version(version: str) -> str:
    """将版本增加 0.0.1，并执行 0.6.9 -> 0.7.0 式十进制进位。"""

    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"版本号必须是非负整数三段式格式，实际为：{version}")

    major, minor, patch = map(int, parts)
    patch += 1
    minor += patch // 10
    patch %= 10
    major += minor // 10
    minor %= 10
    return f"{major}.{minor}.{patch}"


def read_version(version_file: Path) -> str:
    """读取版本文件中的唯一版本字段。"""

    content = version_file.read_text(encoding="utf-8")
    matches = list(VERSION_PATTERN.finditer(content))
    if len(matches) != 1:
        raise RuntimeError(
            f"{version_file} 中应恰好存在一个版本字段，实际找到 {len(matches)} 个。"
        )
    return matches[0].group("version")


def write_version(version_file: Path, version: str) -> None:
    """只替换目标版本字段，并保留文件中的所有其他内容。"""

    content = version_file.read_text(encoding="utf-8")
    updated, count = VERSION_PATTERN.subn(
        lambda match: f"{match.group('prefix')}{version}{match.group('suffix')}",
        content,
    )
    if count != 1:
        raise RuntimeError(f"版本字段应更新一次，实际更新 {count} 次。")
    version_file.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="递增 Python 包的十进制版本")
    parser.add_argument("--base-version", help="递增基准；CI 中传入 PyPI 当前版本。")
    parser.add_argument(
        "--version-file",
        type=Path,
        default=project_root / "setup.py",
        help="版本文件路径，默认是项目根目录的 setup.py。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出下一版本，不写文件。")
    return parser.parse_args()


def main() -> None:
    """计算下一版本，并在非 dry-run 模式下更新版本文件。"""

    args = parse_args()
    version_file = args.version_file.resolve()
    base_version = args.base_version or read_version(version_file)
    next_version = increment_version(base_version)
    if not args.dry_run:
        write_version(version_file, next_version)
    print(next_version)


if __name__ == "__main__":
    main()
