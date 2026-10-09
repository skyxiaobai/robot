# -*- coding: utf-8 -*-
"""四级标注校验：ENVIRONMENT、TASK、时间分段 SUBTASK、分手 INSTRUCTION。"""
_HANDS = ("left", "right", "both")
_TIME_TOL = 1e-3


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _check_coverage(segments, duration_s, label):
    errors = []
    if not segments:
        return errors
    ordered = sorted(segments, key=lambda item: item["t_start"])
    if abs(ordered[0]["t_start"]) > _TIME_TOL:
        errors.append("%s 必须从 0 开始" % label)
    for previous, current in zip(ordered, ordered[1:]):
        if current["t_start"] - previous["t_end"] > _TIME_TOL:
            errors.append("%s 在 %.3f 与 %.3f 之间有空档" % (label, previous["t_end"], current["t_start"]))
        if previous["t_end"] - current["t_start"] > _TIME_TOL:
            errors.append("%s 在 %.3f 处重叠" % (label, current["t_start"]))
    if duration_s is not None and abs(ordered[-1]["t_end"] - float(duration_s)) > _TIME_TOL:
        errors.append("%s 必须覆盖到片段结束 %.3f" % (label, float(duration_s)))
    return errors


def validate_annotation(annotation, duration_s=None, strict=False):
    """校验四级标注。

    非严格模式：结构错误使 ``ok`` 为假；缺 SUBTASK / INSTRUCTION 只记入
    ``missing_levels``。严格模式把缺级也当作失败。
    """
    errors = []
    missing = []
    if not isinstance(annotation, dict):
        return {"ok": False, "errors": ["annotation 必须是对象"], "missing_levels": ["environment", "task", "subtask", "instruction"]}

    environment = annotation.get("environment")
    if not isinstance(environment, dict) or not str(environment.get("name") or "").strip():
        errors.append("ENVIRONMENT.name 不能为空")
    task = annotation.get("task")
    if not isinstance(task, dict) or not str(task.get("name") or "").strip():
        errors.append("TASK.name 不能为空")
    if not isinstance(task, dict) or not str(task.get("instruction") or "").strip():
        errors.append("TASK.instruction 不能为空")

    subtasks = annotation.get("subtasks")
    if not isinstance(subtasks, list):
        errors.append("SUBTASK 必须是数组")
        subtasks = []
    elif len(subtasks) == 0:
        missing.append("subtask")
    parsed_subtasks = []
    for index, segment in enumerate(subtasks):
        if not isinstance(segment, dict):
            errors.append("SUBTASK[%d] 必须是对象" % index)
            continue
        start = _number(segment.get("t_start"))
        end = _number(segment.get("t_end"))
        text = segment.get("text")
        if start is None or end is None or end <= start:
            errors.append("SUBTASK[%d] 的时间区间无效" % index)
            continue
        if not isinstance(text, str) or not text.strip():
            errors.append("SUBTASK[%d] 缺少 text" % index)
        parsed_subtasks.append({"t_start": start, "t_end": end})
    if len(parsed_subtasks) == len(subtasks) and subtasks:
        errors.extend(_check_coverage(parsed_subtasks, duration_s, "SUBTASK"))

    instructions = annotation.get("instructions")
    if not isinstance(instructions, list):
        errors.append("INSTRUCTION 必须是数组")
        instructions = []
    elif len(instructions) == 0:
        missing.append("instruction")
    for index, segment in enumerate(instructions):
        if not isinstance(segment, dict):
            errors.append("INSTRUCTION[%d] 必须是对象" % index)
            continue
        start = _number(segment.get("t_start"))
        end = _number(segment.get("t_end"))
        hand = segment.get("hand")
        text = segment.get("text")
        if start is None or end is None or end <= start:
            errors.append("INSTRUCTION[%d] 的时间区间无效" % index)
        elif duration_s is not None and (start < -_TIME_TOL or end > float(duration_s) + _TIME_TOL):
            errors.append("INSTRUCTION[%d] 超出片段时长" % index)
        if hand not in _HANDS:
            errors.append("INSTRUCTION[%d] 的 hand 必须是 left、right 或 both" % index)
        if not isinstance(text, str) or not text.strip():
            errors.append("INSTRUCTION[%d] 缺少 text" % index)

    ok = not errors and not (strict and missing)
    return {"ok": ok, "errors": errors, "missing_levels": missing}
