"""Convert an isometric dungeon minimap screenshot into a top-down occupancy grid.

Pipeline:
  1. Segment the gray walkable floor with a brightness threshold.
  2. Keep the largest connected floor component (drops the HUD banner, etc).
  3. Find the four extreme points of the floor diamond and perspective-rectify it
     into a square top-down image.
  4. Drop small dark blobs (floor decorations) while keeping thin wall lines.
  5. Seal geometry with a morphological close and build a distance-transform
     clearance field used by the RL environment for collision checks.

Outputs a single ``.npz`` with:
  free       (H, W) uint8, 1 = walkable, 0 = wall
  clearance  (H, W) float32, distance (in cells) to the nearest wall
  resolution int, side length in cells of the square grid
  origin     (2,) float32, world coordinate of pixel (0, 0)
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _find_corners(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("empty floor mask")
    top = np.array([xs[ys.argmin()], ys.min()], dtype=np.float32)
    bottom = np.array([xs[ys.argmax()], ys.max()], dtype=np.float32)
    left = np.array([xs.min(), ys[xs.argmin()]], dtype=np.float32)
    right = np.array([xs.max(), ys[xs.argmax()]], dtype=np.float32)
    return np.stack([top, right, bottom, left])


def build_grid(
    image_path: Path,
    resolution: int = 512,
    floor_threshold: float = 70.0,
    min_wall_area: int = 150,
    close_ksize: int = 3,
) -> dict:
    rgb = np.asarray(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)).astype(np.int32)
    gray = rgb.mean(axis=2)

    floor = (gray > floor_threshold).astype(np.uint8)
    floor = _largest_component(floor)
    corners = _find_corners(floor)

    side = resolution - 1
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        floor * 255, transform, (resolution, resolution), flags=cv2.INTER_NEAREST
    )
    free = (warped > 0).astype(np.uint8)
    wall = 1 - free

    # Remove small blobs (decorative dots on the floor) but keep long wall lines.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64).copy()
    areas[0] = np.iinfo(np.int64).max
    small = np.isin(labels, np.where(areas < min_wall_area)[0])
    wall[small] = 0

    if close_ksize > 0:
        kernel = np.ones((close_ksize, close_ksize), np.uint8)
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel)

    free = (1 - wall).astype(np.uint8)
    free = _largest_component(free)

    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)

    return {
        "free": free,
        "clearance": clearance,
        "resolution": np.int32(resolution),
        "corners": corners.astype(np.float32),
    }


def build_goal_fields(
    clearance: np.ndarray,
    coarse_factor: int = 4,
    num_goals: int = 256,
    min_clearance: float = 4.0,
    seed: int = 0,
) -> dict:
    """Downsample traversability and precompute a coarse geodesic field per goal.

    The geodesic fields give the RL reward a non-deceptive potential: progress
    is measured along the maze instead of straight-line distance, so the policy
    is not lured into dead ends.
    """
    h, w = clearance.shape
    hc, wc = h // coarse_factor, w // coarse_factor
    trav_full = clearance >= min_clearance
    trav = trav_full.reshape(hc, coarse_factor, wc, coarse_factor).any(axis=(1, 3))

    # Restrict goals to the largest coarse connected component.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(trav.astype(np.uint8), 4)
    if len(stats) <= 1:
        raise ValueError("coarse traversable mask is empty")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    trav = labels == main

    rng = np.random.default_rng(seed)
    rows, cols = np.where(trav)
    order = rng.permutation(len(rows))[:num_goals]
    goal_rc = np.stack([rows[order], cols[order]], axis=1).astype(np.int64)

    def bfs(src):
        dist = np.full((hc, wc), 1.0e6, np.float32)
        dist[src] = 0.0
        q = deque([tuple(src)])
        while q:
            r, c = q.popleft()
            d = dist[r, c] + 1.0
            if r + 1 < hc and trav[r + 1, c] and dist[r + 1, c] > d:
                dist[r + 1, c] = d
                q.append((r + 1, c))
            if r - 1 >= 0 and trav[r - 1, c] and dist[r - 1, c] > d:
                dist[r - 1, c] = d
                q.append((r - 1, c))
            if c + 1 < wc and trav[r, c + 1] and dist[r, c + 1] > d:
                dist[r, c + 1] = d
                q.append((r, c + 1))
            if c - 1 >= 0 and trav[r, c - 1] and dist[r, c - 1] > d:
                dist[r, c - 1] = d
                q.append((r, c - 1))
        return dist

    fields = np.empty((len(goal_rc), hc, wc), np.float32)
    for i, (r, c) in enumerate(goal_rc):
        fields[i] = bfs((r, c))

    # Pick the most central full-res cell inside each goal cell as the target position.
    goal_pixel = np.zeros((len(goal_rc), 2), np.int32)
    for i, (r, c) in enumerate(goal_rc):
        block = clearance[
            r * coarse_factor : (r + 1) * coarse_factor,
            c * coarse_factor : (c + 1) * coarse_factor,
        ]
        br, bc = np.unravel_index(int(np.argmax(block)), block.shape)
        goal_pixel[i] = (r * coarse_factor + br, c * coarse_factor + bc)

    return {
        "coarse_trav": trav.astype(np.uint8),
        "coarse_geo": fields,
        "goal_coarse": goal_rc.astype(np.int32),
        "goal_pixel": goal_pixel,
        "coarse_factor": np.int32(coarse_factor),
    }


def save_preview(grid: dict, path: Path) -> None:
    free = grid["free"]
    vis = np.full((*free.shape, 3), 255, np.uint8)
    vis[free == 0] = 0
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a top-down occupancy grid from a map image")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(Path.home() / "图片" / "大地图3.png"),
        help="input isometric minimap image",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).resolve().parent / "maze_grid_traced.npz"),
        help="output .npz path (traced reference layout)",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--floor-threshold", type=float, default=70.0)
    parser.add_argument("--min-wall-area", type=int, default=150)
    parser.add_argument("--coarse-factor", type=int, default=4)
    parser.add_argument("--num-goals", type=int, default=256)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--skip-goals", action="store_true", help="only build the occupancy grid")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    grid = build_grid(
        image_path,
        resolution=args.resolution,
        floor_threshold=args.floor_threshold,
        min_wall_area=args.min_wall_area,
    )
    if not args.skip_goals:
        grid.update(
            build_goal_fields(
                grid["clearance"],
                coarse_factor=args.coarse_factor,
                num_goals=args.num_goals,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **grid)

    preview = Path(args.preview) if args.preview else output.with_name("maze_grid_preview.png")
    save_preview(grid, preview)

    free = grid["free"]
    clearance = grid["clearance"]
    print(f"grid saved       : {output.resolve()}")
    print(f"preview saved    : {preview.resolve()}")
    print(f"resolution       : {grid['resolution']}x{grid['resolution']}")
    print(f"free fraction    : {free.mean():.3f}")
    print(f"clearance max/med: {clearance.max():.1f} / {np.median(clearance[free > 0]):.1f}")


if __name__ == "__main__":
    main()
