from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def obstacle_bounds(env) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    maze_free = getattr(env, "maze_free", None)
    if maze_free is not None:
        h, w = maze_free.shape
        return -w / 2.0, w / 2.0, -h / 2.0, h / 2.0
    if hasattr(env, "tree_centers"):
        t_centers = env.tree_centers
        xs.extend((t_centers[:, 0] - 28.0).tolist() + (t_centers[:, 0] + 28.0).tolist())
        ys.extend((t_centers[:, 1] - 28.0).tolist() + (t_centers[:, 1] + 28.0).tolist())
    if hasattr(env, "mountain_vertices"):
        m_verts = env.mountain_vertices
        xs.extend(m_verts[..., 0].flatten().tolist())
        ys.extend(m_verts[..., 1].flatten().tolist())
    if hasattr(env, "low_centers"):
        l_centers = env.low_centers
        xs.extend((l_centers[:, 0] - 16.0).tolist() + (l_centers[:, 0] + 16.0).tolist())
        ys.extend((l_centers[:, 1] - 16.0).tolist() + (l_centers[:, 1] + 16.0).tolist())
    if not xs:
        half = getattr(env, "world_size", 1000.0) / 2.0
        return -half, half, -half, half
    return min(xs), max(xs), min(ys), max(ys)


def add_stop_range(ax, env, *, compact: bool = False) -> None:
    tolerance = float(getattr(env, "stop_distance_tolerance", 0.0))
    center = (float(env.target[0]), float(env.target[1]))
    outer = float(env.desired_stop_distance) + max(tolerance, 0.0)
    inner = max(float(env.desired_stop_distance) - max(tolerance, 0.0), 0.0)
    if outer > inner:
        ax.add_patch(
            Wedge(
                center,
                outer,
                0.0,
                360.0,
                width=outer - inner,
                facecolor="#f59e0b",
                edgecolor="none",
                alpha=0.16 if compact else 0.20,
                zorder=3,
            )
        )
    line_width = 1.7 if compact else 1.4
    for radius, linestyle, color, alpha in (
        (inner, "-", "#f59e0b", 0.95),
        (float(env.desired_stop_distance), "--", "#dc2626", 0.95),
        (outer, "-", "#f59e0b", 0.95),
    ):
        if radius <= 0.0:
            continue
        ax.add_patch(
            plt.Circle(
                center,
                radius,
                edgecolor=color,
                facecolor="none",
                linestyle=linestyle,
                linewidth=line_width,
                alpha=alpha,
                zorder=4,
            )
        )


def agent_triangle(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    right = np.array([-forward[1], forward[0]], dtype=np.float32)
    center = center.astype(np.float32)
    tip = center + forward * length
    back = center - forward * length * 0.42
    left = back + right * width * 0.5
    right_pt = back - right * width * 0.5
    return np.stack([tip, left, right_pt], axis=0)


def angle_error_to_target(point: np.ndarray, target: np.ndarray, heading: float) -> float:
    delta = target - point
    target_angle = math.atan2(float(delta[1]), float(delta[0]))
    return float(math.atan2(math.sin(target_angle - heading), math.cos(target_angle - heading)))


def render_episode_video(
    env,
    path: np.ndarray,
    total_reward: float,
    output: Path,
    *,
    fps: int = 20,
    headings: np.ndarray | None = None,
    view_scale: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#fbfaf5")
    ax.set_aspect("equal", adjustable="box")

    obs_min_x, obs_max_x, obs_min_y, obs_max_y = obstacle_bounds(env)
    margin = 120.0
    min_x = min(path[:, 0].min(), env.target[0], obs_min_x) - margin
    max_x = max(path[:, 0].max(), env.target[0], obs_max_x) + margin
    min_y = min(path[:, 1].min(), env.target[1], obs_min_y) - margin
    max_y = max(path[:, 1].max(), env.target[1], obs_max_y) + margin
    if view_scale > 1.0:
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        half_w = (max_x - min_x) * 0.5 * view_scale
        half_h = (max_y - min_y) * 0.5 * view_scale
        min_x, max_x = center_x - half_w, center_x + half_w
        min_y, max_y = center_y - half_h, center_y + half_h
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.grid(True, color="#e4dfd2", linewidth=0.6)
    ax.tick_params(labelsize=8, colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    maze_free = getattr(env, "maze_free", None)
    if maze_free is not None:
        h, w = maze_free.shape
        bg = np.empty((h, w, 3), dtype=np.float32)
        bg[maze_free > 0] = (0.90, 0.89, 0.85)
        bg[maze_free == 0] = (0.20, 0.19, 0.22)
        ax.imshow(
            bg,
            extent=[-w / 2.0, w / 2.0, -h / 2.0, h / 2.0],
            origin="upper",
            zorder=0,
            interpolation="nearest",
        )

    view_span = min(max_x - min_x, max_y - min_y)
    marker_size = max(10.0, min(24.0, 17000.0 / max(view_span, 1.0)))
    arrow_len = max(18.0, min(34.0, view_span * 0.018))
    arrow_width = arrow_len * 0.62

    # 绘制树木 (Trees)
    if hasattr(env, "tree_centers"):
        t_centers = env.tree_centers
        for idx in range(len(t_centers)):
            ax.add_patch(
                plt.Circle(
                    (t_centers[idx, 0], t_centers[idx, 1]),
                    getattr(env, "tree_radius", 28.0),
                    facecolor="#4f8f4f",
                    edgecolor="#276749",
                    alpha=0.48,
                    linewidth=0.7,
                    fill=True,
                    zorder=1,
                )
            )
    # 绘制多边形大山 (Mountains)
    if hasattr(env, "mountain_vertices"):
        m_verts = env.mountain_vertices
        for idx in range(len(m_verts)):
            ax.add_patch(
                plt.Polygon(
                    m_verts[idx],
                    closed=True,
                    facecolor="#6b6258",
                    edgecolor="#3f3a34",
                    alpha=0.50,
                    linewidth=0.8,
                    fill=True,
                    zorder=1,
                )
            )
    # 绘制可跳跃矮桩 (Low obstacles)
    if hasattr(env, "low_centers"):
        l_centers = env.low_centers
        for idx in range(len(l_centers)):
            ax.add_patch(
                plt.Circle(
                    (l_centers[idx, 0], l_centers[idx, 1]),
                    getattr(env, "low_radius", 16.0),
                    facecolor="#d97706",
                    edgecolor="#b45309",
                    alpha=0.60,
                    linewidth=0.7,
                    fill=True,
                    zorder=1,
                )
            )
    add_stop_range(ax, env)
    ax.scatter(path[0, 0], path[0, 1], color="#16a34a", s=marker_size, marker="o", zorder=5)
    ax.scatter(env.target[0], env.target[1], color="#dc2626", s=marker_size * 1.15, marker="x", linewidths=1.6, zorder=6)
    trail, = ax.plot([], [], color="#2563eb", linewidth=1.25, alpha=0.88, zorder=5)
    agent = Polygon(
        np.zeros((3, 2), dtype=np.float32),
        closed=True,
        facecolor="#111111",
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=7,
    )
    ax.add_patch(agent)
    ax.text(
        path[0, 0],
        path[0, 1],
        "S",
        color="#166534",
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=6,
    )
    ax.text(
        env.target[0],
        env.target[1],
        "T",
        color="#991b1b",
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=6,
    )
    title = ax.set_title("")

    zoom_radius = max(90.0, float(env.desired_stop_distance) + float(getattr(env, "stop_distance_tolerance", 0.0)) + 55.0)
    target_ax = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=1.0)
    target_ax.set_facecolor("#fffaf0")
    target_ax.set_aspect("equal", adjustable="box")
    target_ax.set_xlim(env.target[0] - zoom_radius, env.target[0] + zoom_radius)
    target_ax.set_ylim(env.target[1] - zoom_radius, env.target[1] + zoom_radius)
    target_ax.grid(True, color="#f1e3c2", linewidth=0.5)
    target_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in target_ax.spines.values():
        spine.set_color("#d97706")
        spine.set_linewidth(1.1)
    add_stop_range(target_ax, env, compact=True)
    target_ax.scatter(env.target[0], env.target[1], color="#dc2626", s=18, marker="x", linewidths=1.5, zorder=6)
    inset_trail, = target_ax.plot([], [], color="#2563eb", linewidth=1.2, alpha=0.9, zorder=5)
    inset_agent = Polygon(
        np.zeros((3, 2), dtype=np.float32),
        closed=True,
        facecolor="#111111",
        edgecolor="#ffffff",
        linewidth=0.7,
        zorder=7,
    )
    target_ax.add_patch(inset_agent)
    target_title = target_ax.set_title("target", fontsize=8, color="#92400e", pad=2)

    repeat_each_step = max(1, int(round(env.dt * fps)))
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-preset", "ultrafast", "-threads", "0", "-pix_fmt", "yuv420p"], metadata={"title": "BlindNav eval"})

    with writer.saving(fig, str(output), dpi=90):
        for frame_idx in range(len(path)):
            trail.set_data(path[: frame_idx + 1, 0], path[: frame_idx + 1, 1])
            heading = float(headings[frame_idx]) if headings is not None and frame_idx < len(headings) else float(env.heading)
            agent.set_xy(agent_triangle(path[frame_idx], heading, arrow_len, arrow_width))
            inset_trail.set_data(path[: frame_idx + 1, 0], path[: frame_idx + 1, 1])
            inset_agent.set_xy(agent_triangle(path[frame_idx], heading, arrow_len * 0.72, arrow_width * 0.72))
            dist = float(np.linalg.norm(env.target - path[frame_idx]))
            stop_err = abs(dist - float(env.desired_stop_distance))
            angle_err = angle_error_to_target(path[frame_idx], env.target, heading)
            angle_err_deg = math.degrees(angle_err)
            dist_ok = stop_err <= float(env.stop_distance_tolerance)
            angle_ok = abs(angle_err) <= float(env.align_angle_tolerance)
            stop_ok = dist_ok
            target_title.set_text(
                f"target zoom | angle {angle_err_deg:+.1f}deg "
                f"({'ok' if angle_ok else 'off'}) | {'STOP' if stop_ok else 'move'}"
            )
            from datetime import datetime
            now_str = datetime.now().strftime("%H:%M:%S")
            title.set_text(
                f"[{now_str} PURE-RL] step={frame_idx} dist={dist:.1f} "
                f"stop_err={stop_err:.1f} "
                f"angle_err={angle_err_deg:+.1f}deg "
                f"STOP={'yes' if stop_ok else 'no'}"
            )
            for _ in range(repeat_each_step):
                writer.grab_frame()
        for _ in range(fps):
            writer.grab_frame()
    plt.close(fig)


def check_video_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    try:
        res = subprocess.run(["ffprobe", "-v", "error", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return True

def concat_and_compress(videos: list[Path], output_concat: Path, output_compressed: Path, *, fps: int) -> None:
    # 过滤不存在、损坏或无 moov atom 的无效 mp4 文件
    valid_videos = [v.resolve() for v in videos if check_video_valid(v)]
    if not valid_videos:
        print("[警告] 没有找到有效的单帧 MP4 视频，跳过拼接。")
        return
    import shutil
    out_comp_abs = output_compressed.resolve()
    out_concat_abs = output_concat.resolve()

    if len(valid_videos) == 1:
        cmd = ["ffmpeg", "-y", "-i", str(valid_videos[0]), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out_comp_abs)]
    else:
        inputs = []
        for v in valid_videos:
            inputs.extend(["-i", str(v)])
        filter_spec = "".join(f"[{i}:v]" for i in range(len(valid_videos))) + f"concat=n={len(valid_videos)}:v=1:a=0[outv]"
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_spec,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(out_comp_abs),
        ]
    subprocess.run(cmd, check=True)
    shutil.copyfile(out_comp_abs, out_concat_abs)
