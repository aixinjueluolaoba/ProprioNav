"""Procedurally generate a clean room-and-corridor maze for RL training.

Instead of tracing the jagged pixels of the reference screenshot, this builds a
crisp, axis-aligned dungeon whose style is inspired by it: large open rooms
separated by thin walls, connected by narrow corridors, with a few interior
blocks. Output uses the same ``.npz`` schema as ``build_maze_grid.py`` so the
maze environment can consume either one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from map_env.build_maze_grid import build_goal_fields


def _bsp_rooms(free, rng, x0, y0, x1, y1, min_leaf, max_room, depth, max_depth):
    w, h = x1 - x0, y1 - y0
    if depth >= max_depth or (w <= max_room and h <= max_room):
        rw = int(rng.integers(max(3, w - 4), w + 1))
        rh = int(rng.integers(max(3, h - 4), h + 1))
        rx = int(rng.integers(x0, x1 - rw + 1))
        ry = int(rng.integers(y0, y1 - rh + 1))
        free[ry : ry + rh, rx : rx + rw] = 1
        return [(rx, ry, rx + rw, ry + rh)]

    if w >= h:
        lo, hi = x0 + min_leaf, x1 - min_leaf
        sx = int(rng.integers(lo, hi + 1))
        return _bsp_rooms(free, rng, x0, y0, sx, y1, min_leaf, max_room, depth + 1, max_depth) + \
               _bsp_rooms(free, rng, sx, y0, x1, y1, min_leaf, max_room, depth + 1, max_depth)

    lo, hi = y0 + min_leaf, y1 - min_leaf
    sy = int(rng.integers(lo, hi + 1))
    return _bsp_rooms(free, rng, x0, y0, x1, sy, min_leaf, max_room, depth + 1, max_depth) + \
           _bsp_rooms(free, rng, x0, sy, x1, y1, min_leaf, max_room, depth + 1, max_depth)


def _carve_corridor(free, a, b, width=1):
    ax, ay = a
    bx, by = b
    lo, hi = sorted((ax, bx))
    free[ay - width // 2 : ay + width // 2 + 1, lo : hi + 1] = 1
    lo, hi = sorted((ay, by))
    free[lo : hi + 1, bx - width // 2 : bx + width // 2 + 1] = 1


def _connect_rooms_mst(free, rooms, rng, corridor_width):
    centers = [((r[0] + r[2]) // 2, (r[1] + r[3]) // 2) for r in rooms]
    n = len(centers)
    connected = [0]
    remaining = set(range(1, n))
    while remaining:
        best = None
        for i in connected:
            for j in remaining:
                d = (centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        _carve_corridor(free, centers[i], centers[j], corridor_width)
        connected.append(j)
        remaining.discard(j)


def _add_interior_blocks(free, rooms, rng, blocks_per_room):
    h, w = free.shape
    for (rx0, ry0, rx1, ry1) in rooms:
        for _ in range(int(rng.integers(0, blocks_per_room + 1))):
            rw = int(rng.integers(1, 4))
            rh = int(rng.integers(1, 4))
            if rx1 - rx0 - 2 * rw < 2 or ry1 - ry0 - 2 * rh < 2:
                continue
            bx = int(rng.integers(rx0 + 1, rx1 - rw))
            by = int(rng.integers(ry0 + 1, ry1 - rh))
            free[by : by + rh, bx : bx + rw] = 0


def generate(
    grid_cells: int = 64,
    tile_px: int = 8,
    seed: int = 0,
    min_room: int = 6,
    max_room: int = 13,
    corridor_width: int = 2,
    blocks_per_room: int = 2,
    goal_factor: int = 1,
    num_goals: int = 256,
) -> dict:
    rng = np.random.default_rng(seed)
    h = w = grid_cells
    free = np.zeros((h, w), np.uint8)

    min_leaf = max(4, min_room // 2 + 2)
    rooms = _bsp_rooms(free, rng, 1, 1, w - 1, h - 1, min_leaf, max_room, 0, 7)
    _connect_rooms_mst(free, rooms, rng, corridor_width)
    _add_interior_blocks(free, rooms, rng, blocks_per_room)
    free[:, 0] = 0
    free[:, -1] = 0
    free[0, :] = 0
    free[-1, :] = 0

    # Keep the largest connected free component so every spawn is reachable.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    free = (labels == main).astype(np.uint8)

    free_px = cv2.resize(
        free * 255, (w * tile_px, h * tile_px), interpolation=cv2.INTER_NEAREST
    )
    free_px = (free_px > 0).astype(np.uint8)
    clearance = cv2.distanceTransform(free_px, cv2.DIST_L2, 5).astype(np.float32)

    grid = {
        "free": free_px,
        "clearance": clearance,
        "resolution": np.int32(w * tile_px),
    }
    grid.update(
        build_goal_fields(clearance, coarse_factor=goal_factor, num_goals=num_goals)
    )
    return grid


def save_preview(grid: dict, path: Path) -> None:
    free = grid["free"]
    vis = np.full((*free.shape, 3), 255, np.uint8)
    vis[free == 0] = 0
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean procedural maze")
    parser.add_argument(
        "-o", "--output", default=str(Path(__file__).resolve().parent / "maze_grid.npz")
    )
    parser.add_argument("--grid-cells", type=int, default=64)
    parser.add_argument("--tile-px", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-room", type=int, default=6)
    parser.add_argument("--max-room", type=int, default=13)
    parser.add_argument("--corridor-width", type=int, default=2)
    parser.add_argument("--num-goals", type=int, default=256)
    parser.add_argument("--preview", default=None)
    args = parser.parse_args()

    grid = generate(
        grid_cells=args.grid_cells,
        tile_px=args.tile_px,
        seed=args.seed,
        min_room=args.min_room,
        max_room=args.max_room,
        corridor_width=args.corridor_width,
        num_goals=args.num_goals,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **grid)
    preview = Path(args.preview) if args.preview else output.with_name("maze_grid_preview.png")
    save_preview(grid, preview)
    print(f"grid saved    : {output.resolve()}")
    print(f"preview saved : {preview.resolve()}")
    print(f"free fraction : {grid['free'].mean():.3f}")


if __name__ == "__main__":
    main()
