#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验四级标注 JSON。文件可以是标注对象，或带 annotation 字段的 episode。

示例:
    python scripts/validate_hierarchy.py examples/hierarchy_annotation.json --strict
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.labels import validate_annotation  # noqa: E402


def _annotation_and_duration(payload, duration):
    if isinstance(payload, dict) and "annotation" in payload and "environment" not in payload:
        annotation = payload["annotation"]
        if duration is None and payload.get("num_frames") and payload.get("fps"):
            duration = float(payload["num_frames"]) / float(payload["fps"])
    else:
        annotation = payload
        if duration is None and isinstance(payload, dict) and payload.get("duration_s") is not None:
            duration = float(payload["duration_s"])
    return annotation, duration


def main(argv=None):
    parser = argparse.ArgumentParser(description="校验 ENVIRONMENT/TASK/SUBTASK/INSTRUCTION")
    parser.add_argument("path", help="标注 JSON 或统一 episode JSON")
    parser.add_argument("--strict", action="store_true", help="缺少 SUBTASK 或 INSTRUCTION 时失败")
    parser.add_argument("--duration", type=float, default=None, help="片段时长（秒），用于检查是否盖满")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    annotation, duration = _annotation_and_duration(payload, args.duration)
    result = validate_annotation(annotation, duration_s=duration, strict=args.strict)
    for item in result["errors"]:
        print("error: " + item)
    if result["missing_levels"]:
        print("missing: " + ", ".join(result["missing_levels"]))
    print("ok" if result["ok"] else "invalid")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
