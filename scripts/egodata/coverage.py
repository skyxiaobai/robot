# -*- coding: utf-8 -*-
"""按环境、物体、任务、动作类型统计覆盖，并标出封闭词表里的空档。"""
import csv
import html
import re
from pathlib import Path

# 封闭词表。任务名和原始物体名是开放词表，只统计出现过的值，不制造空档。
TAXONOMY = {
    "environment": ("tabletop", "kitchen", "living_room", "workshop", "outdoor", "unknown"),
    "object_class": (
        "container",
        "tool",
        "cloth",
        "food",
        "electronics",
        "toy",
        "furniture",
        "tableware",
        "other",
        "unknown",
    ),
    "action_type": (
        "pick",
        "place",
        "open",
        "close",
        "pour",
        "wipe",
        "fold",
        "tie",
        "cut",
        "stir",
        "insert",
        "remove",
        "stack",
        "screw",
        "throw",
        "type",
        "assemble",
        "charge",
        "zip",
        "scoop",
        "play",
        "thread",
        "dry",
        "wrap",
        "roll",
        "push",
        "point",
        "pop",
        "color",
        "other",
        "unknown",
    ),
}

_LABELS = {
    "environment": "环境",
    "object": "物体",
    "object_class": "物体类别",
    "task": "任务",
    "action_type": "动作类型",
    "tabletop": "桌面",
    "kitchen": "厨房",
    "living_room": "起居",
    "workshop": "工作间",
    "outdoor": "户外",
    "unknown": "未知",
    "container": "容器",
    "tool": "工具",
    "cloth": "布料",
    "food": "食物",
    "electronics": "电子",
    "toy": "玩具",
    "furniture": "家具",
    "tableware": "餐具",
    "other": "其他",
    "pick": "抓取",
    "place": "放置",
    "open": "打开",
    "close": "关闭",
    "pour": "倾倒",
    "wipe": "擦拭",
    "fold": "折叠",
    "tie": "打结",
    "cut": "切割",
    "stir": "搅拌",
    "insert": "放入",
    "remove": "取出",
    "stack": "堆叠",
    "screw": "拧转",
    "throw": "抛接",
    "type": "按键",
    "assemble": "组装",
    "charge": "充电",
    "zip": "拉链",
    "scoop": "舀取",
    "play": "摆弄",
    "thread": "穿线",
    "dry": "擦干",
    "wrap": "包裹",
    "roll": "滚动",
    "push": "推动",
    "point": "指向",
    "pop": "按压",
    "color": "涂色",
}

_OBJECT_KEYWORDS = (
    ("tableware", ("cup", "mug", "plate", "bowl", "dish", "utensil", "chopstick", "spoon", "fork", "knife", "tray", "pan", "pot")),
    ("container", ("tupperware", "bottle", "drawer", "basket", "carton", "crate", "case", "box", "jar", "bag", "bin", "lid", "cap", "can", "safe")),
    ("cloth", ("tablecloth", "shoelace", "napkin", "glove", "cloth", "shirt", "towel", "sleeve", "sock", "lace")),
    ("food", ("sandwich", "topping", "vegetable", "banana", "orange", "apple", "bread", "fruit", "food", "rice", "egg", "ice")),
    ("electronics", ("headphone", "controller", "keyboard", "charger", "battery", "monitor", "adapter", "laptop", "tablet", "screen", "airpod", "cable", "mouse", "phone", "remote", "device", "usb")),
    ("tool", ("screwdriver", "toothbrush", "scissors", "sponge", "pencil", "marker", "crayon", "broom", "brush", "tape", "tool", "plug", "pen", "key")),
    ("toy", ("puzzle", "plush", "slime", "chess", "block", "piano", "cube", "bead", "doll", "lego", "dice", "card", "ball", "toy")),
    ("furniture", ("furniture", "shelf", "table", "chair", "desk", "stool")),
)

_ACTION_KEYWORDS = (
    ("pick", ("pick", "grab", "grasp", "gather", "take")),
    ("place", ("place", "put", "set", "stock", "add", "reset")),
    ("open", ("open", "unlock")),
    ("close", ("close", "lock")),
    ("pour", ("pour", "dump", "dispense")),
    ("wipe", ("wipe", "clean", "sweep", "wash")),
    ("fold", ("fold", "unfold")),
    ("tie", ("tie", "untie", "braid", "knot")),
    ("cut", ("cut", "slice", "chop")),
    ("stir", ("stir", "mix", "knead")),
    ("insert", ("insert", "load", "slot", "plug")),
    ("remove", ("remove", "extract", "unplug", "unstock")),
    ("stack", ("stack", "unstack")),
    ("screw", ("screw", "unscrew")),
    ("throw", ("throw", "catch", "toss")),
    ("type", ("type", "click", "press")),
    ("assemble", ("disassemble", "assemble")),
    ("charge", ("uncharge", "charge", "discharge")),
    ("zip", ("unzip", "zip")),
    ("scoop", ("scoop", "ladle")),
    ("play", ("play",)),
    ("thread", ("thread", "unthread")),
    ("dry", ("dry",)),
    ("wrap", ("unwrap", "wrap")),
    ("roll", ("roll", "unroll")),
    ("push", ("push",)),
    ("point", ("point",)),
    ("pop", ("pop",)),
    ("color", ("color", "colour")),
)

# 复数和常见别名。先去掉空格再查，使 "square table" 与 "squaretable" 相同。
_OBJECT_ALIASES = {
    "plates": "plate",
    "cups": "cup",
    "glasses": "glass",
    "boxes": "box",
    "cases": "case",
    "plushie": "plush",
    "plushies": "plush",
    "plushy": "plush",
}

# tablecloth 必须写在 table 前面，否则 table 会先吃掉 tablecloth。
_ENV_FIELD = re.compile(r"(tablecloth|table|position|background)\s*:\s*([^,;]+)")


def _label(value):
    return _LABELS.get(value, value)


def _room_keyword(raw):
    if any(token in raw for token in ("kitchen", "fridge", "sink", "stove")):
        return "kitchen"
    if any(token in raw for token in ("workshop", "garage")):
        return "workshop"
    if any(token in raw for token in ("outdoor", "garden", "yard")):
        return "outdoor"
    if any(token in raw for token in ("sofa", "couch", "living")):
        return "living_room"
    return None


def _canon_env_value(value):
    """去掉空白，并把拼写变体收成同一个颜色名。"""
    token = re.sub(r"\s+", "", (value or "").strip().lower())
    return token.replace("lavendar", "lavender")


def normalize_environment(text):
    """房间词表优先。EgoDex 的桌面、桌布、坐姿和背景颜色都保留。

    ``tablecloth:`` 是桌布颜色，不能被 ``table`` 前缀吃掉。
    ``lavendar`` 与 ``lavender`` 算同一种颜色。
    """
    raw = (text or "").strip().lower()
    if not raw:
        return "unknown"
    fields = {}
    for match in _ENV_FIELD.finditer(raw):
        fields[match.group(1)] = _canon_env_value(match.group(2))
    if fields:
        room = _room_keyword(raw)
        if room:
            return room
        parts = ["tabletop"]
        for key in ("table", "tablecloth", "position", "background"):
            if fields.get(key):
                parts.append("%s=%s" % (key, fields[key]))
        return "|".join(parts)
    room = _room_keyword(raw)
    if room:
        return room
    if any(token in raw for token in ("tabletop", "desk", "table")):
        return "tabletop"
    return "unknown"


def normalize_object_name(name):
    """去掉空格，并把常见复数、别名收成同一个物体名。"""
    raw = re.sub(r"[\s_\-]+", "", (name or "").strip().lower())
    if not raw:
        return "unknown"
    if raw in _OBJECT_ALIASES:
        return _OBJECT_ALIASES[raw]
    if len(raw) > 4 and raw.endswith("s") and not raw.endswith("ss"):
        stem = raw[:-1]
        return _OBJECT_ALIASES.get(stem, stem)
    return raw


def coarse_object_class(name):
    """最长关键词优先，避免 ice 把 device、pen 把别的更长名字抢走。"""
    raw = normalize_object_name(name)
    if raw == "unknown":
        return "unknown"
    best_label = None
    best_length = -1
    for label, words in _OBJECT_KEYWORDS:
        for word in words:
            token = word.replace(" ", "")
            if token and token in raw and len(token) > best_length:
                best_label = label
                best_length = len(token)
    if best_label:
        return best_label
    return "other"


def normalize_action(verb):
    """最长关键词优先，避免 unplug 被 plug、disassemble 被更短的词抢走。"""
    raw = (verb or "").lower().strip().replace("_", " ").replace("-", " ")
    if not raw:
        return "unknown"
    tokens = raw.split()
    compact = "".join(tokens)
    best_length = -1
    best_label = None
    for label, words in _ACTION_KEYWORDS:
        for word in words:
            word_compact = word.replace(" ", "")
            matched = word in tokens or word_compact == compact
            if not matched and len(word_compact) >= 5 and word_compact in compact:
                matched = True
            if matched and len(word_compact) > best_length:
                best_length = len(word_compact)
                best_label = label
    if best_label:
        return best_label
    return "other"


def _episode_axes(episode):
    coverage = episode.get("coverage") or {}
    environment = coverage.get("environment") or "unknown"
    objects = []
    seen_objects = set()
    for name in coverage.get("objects") or []:
        normalized = normalize_object_name(name)
        if normalized not in seen_objects:
            seen_objects.add(normalized)
            objects.append(normalized)
    # 类别按当前词表从物体名重算，这样补关键词不必先重转 JSON。
    if objects:
        classes = []
        for name in objects:
            mapped = coarse_object_class(name)
            if mapped not in classes:
                classes.append(mapped)
    else:
        classes = list(coverage.get("object_classes") or []) or ["unknown"]
    task = coverage.get("task") or (episode.get("annotation") or {}).get("task", {}).get("name") or "unknown"
    actions = list(coverage.get("action_types") or []) or ["unknown"]
    frames = int(episode.get("num_frames") or 0)
    return {
        "environment": [environment],
        "object": objects or ["unknown"],
        "object_class": classes,
        "task": [task],
        "action_type": actions,
        "frames": frames,
    }


def accumulate_coverage(buckets, episode):
    """把一条 episode 累加进计数。调用方可以马上丢掉这条 JSON。"""
    axes = _episode_axes(episode)
    frames = axes["frames"]
    for axis in ("environment", "object", "object_class", "task", "action_type"):
        for value in axes[axis]:
            key = (axis, value)
            bucket = buckets.setdefault(key, {"episodes": 0, "frames": 0})
            bucket["episodes"] += 1
            bucket["frames"] += frames
    return buckets


def finalize_coverage(buckets, n_episodes):
    """由累加结果写出报告，并补上封闭词表里的空档。"""
    counts = []
    for (axis, value), bucket in sorted(buckets.items()):
        taxonomy = TAXONOMY.get(axis)
        counts.append({
            "axis": axis,
            "value": value,
            "episodes": bucket["episodes"],
            "frames": bucket["frames"],
            "in_taxonomy": True if taxonomy is None else value in taxonomy,
        })
    gaps = []
    for axis, values in TAXONOMY.items():
        present = {row["value"] for row in counts if row["axis"] == axis and row["episodes"] > 0}
        for value in values:
            filled = value in present or any(
                item == value or item.startswith(value + "|") for item in present
            )
            if not filled:
                gaps.append({"axis": axis, "value": value})
                counts.append({
                    "axis": axis,
                    "value": value,
                    "episodes": 0,
                    "frames": 0,
                    "in_taxonomy": True,
                })
    counts.sort(key=lambda row: (row["axis"], -row["episodes"], row["value"]))
    return {"episodes": int(n_episodes), "counts": counts, "gaps": gaps}


def coverage_report(episodes):
    """统计各轴取值的片段数，并列出封闭词表中计数为 0 的空档。"""
    buckets = {}
    for episode in episodes:
        accumulate_coverage(buckets, episode)
    return finalize_coverage(buckets, len(episodes))


def write_coverage_reports(report, html_path, csv_path):
    html_path = Path(html_path)
    csv_path = Path(csv_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    current = None
    for row in report["counts"]:
        if row["axis"] != current:
            if current is not None:
                body.append("</table>")
            current = row["axis"]
            body.append("<h2>%s</h2><table><tr><th>取值</th><th>片段</th><th>帧</th></tr>" % html.escape(_label(current)))
        mark = " class=\"gap\"" if row["episodes"] == 0 else ""
        body.append(
            "<tr%s><td>%s</td><td>%d</td><td>%d</td></tr>"
            % (mark, html.escape(_label(row["value"])), row["episodes"], row["frames"])
        )
    if current is not None:
        body.append("</table>")
    document = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>覆盖度</title>
<style>
body { font-family: sans-serif; margin: 2rem; }
table { border-collapse: collapse; margin-bottom: 1.5rem; }
td, th { border: 1px solid #ccc; padding: 0.4rem 0.6rem; }
tr.gap td { color: #a40; }
</style></head><body>
<h1>采集覆盖</h1>
<p>共 %d 条片段。计数为 0 的行是封闭词表里的空档，采集时优先补这些组合。</p>
%s
</body></html>
""" % (report["episodes"], "\n".join(body))
    html_path.write_text(document, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "value", "episodes", "frames", "in_taxonomy", "gap"])
        for row in report["counts"]:
            writer.writerow([
                row["axis"],
                row["value"],
                row["episodes"],
                row["frames"],
                "yes" if row["in_taxonomy"] else "no",
                "yes" if row["episodes"] == 0 else "no",
            ])
    return html_path, csv_path
