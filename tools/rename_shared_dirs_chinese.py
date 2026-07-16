#!/usr/bin/env python3
"""Rename generated shared-artifact directories to Chinese names."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


TOKEN_MAP = {
    "a2c": "优势演员评论家",
    "angle": "角度",
    "annotation": "标注",
    "aspect": "比例",
    "avatar": "头像",
    "benchmark": "基准测试",
    "best": "最佳",
    "better": "优化",
    "blind": "盲人",
    "box": "框",
    "candidate": "候选",
    "clean": "清理版",
    "clip": "片段",
    "clips": "片段",
    "collisiontime": "碰撞时间",
    "compare": "对比",
    "concat": "拼接",
    "concave": "内凹",
    "contact": "接触",
    "current": "当前",
    "cv": "视觉",
    "dataset": "数据集",
    "datasets": "数据集",
    "ddpg": "深度确定策略梯度",
    "delta": "增量",
    "demo": "演示",
    "deploy": "部署",
    "diagnostic": "诊断",
    "diagnostics": "诊断",
    "dist": "距离",
    "dt": "步长",
    "ep": "轮",
    "eval": "测试",
    "even": "均匀",
    "features": "特征",
    "final": "最终",
    "fixed": "固定",
    "frame": "帧",
    "frames": "帧",
    "from": "来自",
    "game": "游戏",
    "harder": "复杂",
    "heatmap": "热力图",
    "hero": "英雄",
    "id": "编号",
    "image": "图片",
    "images": "图片",
    "jumpfix": "跳跃修复",
    "kpl": "王者赛事",
    "large": "大型",
    "line": "直线",
    "lstm": "长短期记忆",
    "map": "映射",
    "marscode": "火星代码",
    "matched": "匹配",
    "medium": "中型",
    "merged": "合并",
    "mlp": "多层感知机",
    "model": "模型",
    "models": "模型",
    "more": "更多",
    "nav": "寻路",
    "no": "无",
    "observable": "可观测",
    "obstacles": "障碍",
    "offset": "偏移",
    "only": "仅",
    "overlay": "叠加",
    "pack": "包",
    "pipeline": "流程",
    "pillow": "渲染",
    "ppo": "近端策略优化",
    "preview": "预览",
    "quality": "质量",
    "random": "随机",
    "raw": "原始",
    "real": "真实",
    "resize": "缩放",
    "reward": "奖励",
    "roi": "兴趣区",
    "rppo": "循环近端策略优化",
    "sac": "软演员评论家",
    "sam": "分割模型",
    "sample": "样本",
    "samples": "样本",
    "search": "搜索",
    "seed": "随机种子",
    "selection": "选择",
    "siam": "孪生跟踪",
    "siamfc": "孪生全卷积跟踪",
    "small": "小型",
    "snap": "截图",
    "stage": "阶段",
    "state": "状态",
    "stop": "停止",
    "sweep": "扫描",
    "target": "目标",
    "td3": "双延迟确定策略",
    "template": "模板",
    "test": "测试",
    "tiny": "极小",
    "tools": "工具",
    "uncompressed": "未压缩",
    "v": "版本",
    "vec": "向量",
    "video": "视频",
    "work": "工作区",
    "wzry": "王者荣耀",
    "yolo": "目标检测",
}


ASCII_RE = re.compile(r"[A-Za-z]")
PART_RE = re.compile(r"[A-Za-z]+|\d+x\d+|\d+|[^A-Za-z\d_]+")


def has_ascii(text: str) -> bool:
    return bool(ASCII_RE.search(text))


def number_word(value: str) -> str:
    return value.replace("x", "乘")


def translate_part(part: str) -> str:
    lower = part.lower()
    if not lower:
        return ""
    if lower.isdigit():
        return lower
    if re.fullmatch(r"\d+x\d+", lower):
        return number_word(lower)
    if match := re.fullmatch(r"(\d+)k", lower):
        return f"{int(match.group(1)) // 10 if int(match.group(1)) >= 10 else match.group(1)}万轮"
    if match := re.fullmatch(r"(\d+)ep", lower):
        return f"{match.group(1)}轮"
    if match := re.fullmatch(r"v(\d+)", lower):
        return f"版本{match.group(1)}"
    if match := re.fullmatch(r"dt(\d+)", lower):
        return f"步长{match.group(1)}"
    if match := re.fullmatch(r"vec(\d+)", lower):
        return f"向量{match.group(1)}"
    if match := re.fullmatch(r"eval(\d+)", lower):
        return f"测试{match.group(1)}"
    if match := re.fullmatch(r"clip(\d+)", lower):
        return f"片段{match.group(1)}"
    if match := re.fullmatch(r"frame(\d+)", lower):
        return f"帧{match.group(1)}"
    if match := re.fullmatch(r"lstm(\d+(?:x\d+)?)", lower):
        return f"长短期记忆{number_word(match.group(1))}"
    if match := re.fullmatch(r"mlp(\d+(?:x\d+)?)", lower):
        return f"多层感知机{number_word(match.group(1))}"
    if lower in TOKEN_MAP:
        return TOKEN_MAP[lower]
    pieces = PART_RE.findall(lower)
    translated = "".join(TOKEN_MAP.get(piece, piece) for piece in pieces)
    if has_ascii(translated):
        return f"目录{abs(hash(part)) % 100000}"
    return translated


def to_chinese_name(name: str) -> str:
    if name.startswith("."):
        return name
    parts = [translate_part(part) for part in re.split(r"[_\-\s]+", name) if part]
    result = "－".join(part for part in parts if part)
    if not result:
        result = "目录"
    if has_ascii(result):
        result = f"目录{abs(hash(name)) % 100000}"
    return result


def unique_target(path: Path, new_name: str) -> Path:
    target = path.with_name(new_name)
    if not target.exists() or target == path:
        return target
    index = 2
    while True:
        candidate = path.with_name(f"{new_name}（{index}）")
        if not candidate.exists():
            return candidate
        index += 1


def collect_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for current, subdirs, _files in os.walk(root):
            subdirs[:] = [name for name in subdirs if not name.startswith(".")]
            current_path = Path(current)
            if current_path != root and has_ascii(current_path.name):
                dirs.append(current_path)
    return sorted(dirs, key=lambda path: len(path.parts), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("roots", nargs="*", default=["/home/diana/screencap/file", "/tmp/file"])
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    renames = []
    for path in collect_dirs(roots):
        target = unique_target(path, to_chinese_name(path.name))
        if target != path:
            renames.append((path, target))

    for old, new in renames[:300]:
        print(f"{old} -> {new}")
    if len(renames) > 300:
        print(f"... omitted {len(renames) - 300} more")
    print(f"total_dirs_to_rename={len(renames)}")

    if args.apply:
        for old, new in renames:
            if old.exists():
                old.rename(new)
        print("applied=1")
    else:
        print("applied=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
