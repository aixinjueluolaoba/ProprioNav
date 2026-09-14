"""Render maze rollouts to images / video for quick visual inspection."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def base_image(free: np.ndarray, scale: int = 2, grid_lines: bool = False) -> np.ndarray:
    """Return an RGB image (already upscaled) with walkable cells light."""
    h, w = free.shape
    img = np.full((h, w, 3), (30, 30, 36), np.uint8)
    img[free > 0] = (235, 235, 228)
    if grid_lines:
        img[::16, :] = (200, 200, 200)
        img[:, ::16] = (200, 200, 200)
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return img


def world_to_pixel(points: np.ndarray, grid_shape: tuple[int, int], scale: int) -> np.ndarray:
    h, w = grid_shape
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    col = (points[:, 0] + w / 2.0) * scale
    row = (h / 2.0 - points[:, 1]) * scale
    return np.stack([col, row], axis=1)


def draw_rollout(
    free: np.ndarray,
    path: np.ndarray,
    target: np.ndarray,
    headings: np.ndarray | None = None,
    scale: int = 2,
    title: str | None = None,
) -> np.ndarray:
    img = base_image(free, scale=scale)
    pts = world_to_pixel(path, free.shape, scale).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (230, 60, 60), max(1, scale), cv2.LINE_AA)

    start_px = tuple(pts[0])
    cv2.circle(img, start_px, 5 * scale // 2 + 2, (30, 160, 60), -1, cv2.LINE_AA)
    tgt_px = tuple(world_to_pixel(target, free.shape, scale)[0].astype(int))
    cv2.circle(img, tgt_px, 4 * scale, (60, 90, 230), -1, cv2.LINE_AA)
    cv2.circle(img, tgt_px, 4 * scale, (255, 255, 255), 1, cv2.LINE_AA)

    if headings is not None:
        headings = np.asarray(headings).reshape(-1)
        step = max(1, len(headings) // 24)
        for i in range(0, min(len(headings), len(pts)), step):
            x, y = pts[i]
            dx = int(round(np.cos(headings[i]) * 3 * scale))
            dy = int(round(-np.sin(headings[i]) * 3 * scale))
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), (40, 40, 40), 1, cv2.LINE_AA, tipLength=0.4)

    if title:
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_rollout_png(
    free: np.ndarray,
    path: np.ndarray,
    target: np.ndarray,
    output: str | Path,
    headings: np.ndarray | None = None,
    scale: int = 2,
    title: str | None = None,
) -> Path:
    img = draw_rollout(free, path, target, headings=headings, scale=scale, title=title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output
