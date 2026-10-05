#!/usr/bin/env python3
"""
pyTri: Python driver for Triangle (https://www.cs.cmu.edu/~quake/triangle.html)

Generates triangular meshes for geophysical domains, refined iteratively
using bathymetry, for hydrodynamic models such as Swash, Swan and Thetis.

Original author: Shuaib Rasheed (30/08/2021)

Modified for large bathymetry files (RegularGridInterpolator + binning) (04/10/2026).

Push minimum angles as high as possible (05/10/2026).

# Example use : python pyTri5.py --file_name TEST --bathy_file_name bath.csv --elevation --bathy_res 80 --num_iterrations 5 --elem_start_size 1500 --elem_end_size 50 --grade 0.30 --shape 1.1 --min_angle 28

"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d

log = logging.getLogger("pyTri")
EQUILATERAL = np.sqrt(3.0) / 4.0


def run_triangle(switches, target):
    cmd = ["triangle", switches, str(target)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Triangle failed:\n%s\n%s", result.stdout, result.stderr)
        raise RuntimeError("Triangle failed")
    log.debug(result.stdout)


def load_mesh(prefix):
    node_data = np.loadtxt(f"{prefix}.node", skiprows=1, usecols=(0, 1, 2), ndmin=2)
    tris = np.loadtxt(f"{prefix}.ele", skiprows=1, usecols=(1, 2, 3), dtype=int, ndmin=2)
    base = int(node_data[0, 0])
    return node_data[:, 1:3], tris - base


def mean_edge_lengths(nodes, tris):
    p = nodes[tris]
    d01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    d12 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    d20 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    return (d01 + d12 + d20) / 3.0


def triangle_angles(nodes, tris):
    """Return min and max angle (degrees) of every triangle."""
    p = nodes[tris]
    a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    b = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)

    cosA = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1, 1)
    cosB = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1, 1)
    cosC = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1, 1)

    angles = np.degrees(np.arccos(np.stack([cosA, cosB, cosC], axis=1)))
    return angles.min(axis=1), angles.max(axis=1)


def report_quality(prefix):
    nodes, tris = load_mesh(prefix)
    min_ang, max_ang = triangle_angles(nodes, tris)
    edges = mean_edge_lengths(nodes, tris)

    log.info("─" * 50)
    log.info("Quality report for %s", prefix)
    log.info("  Triangles          : %d", len(tris))
    log.info("  Edge min/median/max: %.1f / %.1f / %.1f m",
             edges.min(), np.median(edges), edges.max())
    log.info("  Min angle          : %.1f°   (SWAN wants ≥ 45°)", min_ang.min())
    log.info("  Mean min-angle     : %.1f°", min_ang.mean())
    log.info("  Max angle          : %.1f°   (SWAN wants ≤ 135°)", max_ang.max())
    log.info("  %% triangles < 45°  : %.1f%%", 100 * np.mean(min_ang < 45))
    log.info("─" * 50)
    return min_ang.min()


def write_area_file(prefix, areas):
    index = np.arange(1, len(areas) + 1)
    with open(f"{prefix}.area", "w") as f:
        f.write(f"{len(areas)}\n")
        np.savetxt(f, np.column_stack([index, areas]), fmt=["%d", "%.6f"])


def make_bathy_interpolator(bathy_file, resolution=50.0, elevation=False):
    log.info("Loading bathymetry from %s ...", bathy_file)
    df = pd.read_csv(bathy_file, header=0)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    z = df.iloc[:, 2].to_numpy(dtype=float)

    dx = dy = float(resolution)
    x_edges = np.arange(x.min(), x.max() + dx, dx)
    y_edges = np.arange(y.min(), y.max() + dy, dy)

    Z, _, _, _ = binned_statistic_2d(x, y, z, statistic="mean", bins=[x_edges, y_edges])
    mask = np.isnan(Z)
    if mask.any():
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        Z = Z[tuple(ind)]

    X = (x_edges[:-1] + x_edges[1:]) * 0.5
    Y = (y_edges[:-1] + y_edges[1:]) * 0.5
    interp = RegularGridInterpolator((Y, X), Z.T, bounds_error=False, fill_value=None)

    def depth_at(points):
        d = interp(points[:, ::-1])
        return -d if elevation else d

    return depth_at


def limit_gradient(nodes, tris, h, grade, max_iter=400):
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    i, j = edges[:, 0], edges[:, 1]
    dist = np.hypot(*(nodes[i] - nodes[j]).T)
    h = h.copy()
    for _ in range(max_iter):
        before = h.copy()
        np.minimum.at(h, j, h[i] + grade * dist)
        np.minimum.at(h, i, h[j] + grade * dist)
        if np.array_equal(h, before):
            break
    return h


def make_size_fn(h_min, h_max, land_h, d_shallow, d_deep, shape, factor):
    def size_fn(depth):
        s = np.clip((depth - d_shallow) / (d_deep - d_shallow), 0.0, 1.0) ** shape
        h = h_min + (h_max - h_min) * s
        h = np.clip(h * factor, h_min, h_max)
        return np.where(depth <= 0, land_h, h)
    return size_fn


def triangle_areas(nodes, tris, depth_at, size_fn, grade):
    h_node = size_fn(depth_at(nodes))
    h_node = limit_gradient(nodes, tris, h_node, grade)
    centroids = nodes[tris].mean(axis=1)
    h_cent = size_fn(depth_at(centroids))
    h_tri = np.minimum(h_node[tris].min(axis=1), h_cent)
    return EQUILATERAL * h_tri**2, h_tri


def plot_mesh(prefix, out_png):
    nodes, tris = load_mesh(prefix)
    fig, ax = plt.subplots(figsize=(10, 9))
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    size = mean_edge_lengths(nodes, tris)
    pc = ax.tripcolor(tri, facecolors=size, cmap="viridis_r", edgecolors="k", linewidth=0.12)
    pc.set_clim(*np.percentile(size, [2, 98]))
    fig.colorbar(pc, ax=ax, shrink=0.8, label="mean edge length (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{Path(prefix).name}: {len(tris)} triangles")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)


@click.command()
@click.option("--file_name", default="TEST")
@click.option("--bathy_file_name", default="bath.csv")
@click.option("--bathy_res", default=50.0, type=float)
@click.option("--num_iterrations", default=7, type=int, help="Recommend 6–8")
@click.option("--elem_start_size", default=500.0, type=float, help="Coarsest size (m)")
@click.option("--elem_end_size", default=40.0, type=float, help="Finest size (m)")
@click.option("--depth_shallow", default=None, type=float)
@click.option("--depth_deep", default=None, type=float)
@click.option("--shape", default=1.4, type=float, help=">1 keeps fine sizes longer")
@click.option("--grade", default=0.15, type=float, help="Lower = smoother (0.12–0.18 good)")
@click.option("--elevation", is_flag=True)
@click.option("--convex_hull", is_flag=True)
@click.option("--land", "land_mode", type=click.Choice(["coarse", "fine"]), default="coarse")
@click.option("--min_angle", default=30.0, type=float, help="Ask Triangle for 30° (max practical)")
@click.option("--plot/--no-plot", default=True)
@click.option("-v", "--verbose", is_flag=True)
def iter_mesh(file_name, bathy_file_name, bathy_res, num_iterrations,
              elem_start_size, elem_end_size, depth_shallow, depth_deep,
              shape, grade, elevation, convex_hull, land_mode, min_angle, plot, verbose):

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if shutil.which("triangle") is None:
        sys.exit("Triangle not found in PATH")
    if not Path(f"{file_name}.poly").is_file():
        sys.exit(f"Cannot find {file_name}.poly")

    n_steps = num_iterrations - 2
    factors = np.geomspace(2.2, 1.0, n_steps)

    depth_at = make_bathy_interpolator(bathy_file_name, bathy_res, elevation)
    land_h = elem_start_size if land_mode == "coarse" else elem_end_size

    # Highest practical quality
    q = f"q{min_angle:g}e"

    # Initial coarse mesh
    run_triangle(f"-p{'c' if convex_hull else ''}{q}", f"{file_name}.poly")
    run_triangle(f"-rp{q}a{EQUILATERAL * elem_start_size**2:.6f}", f"{file_name}.1")

    for step in range(n_steps):
        prefix = f"{file_name}.{step + 2}"
        nodes, tris = load_mesh(prefix)

        if depth_shallow is None or depth_deep is None:
            d = depth_at(nodes[tris].mean(axis=1))
            d = d[d > 0]
            lo, hi = np.percentile(d, [8, 92])
            depth_shallow = lo if depth_shallow is None else depth_shallow
            depth_deep = hi if depth_deep is None else depth_deep

        if step == 0:
            log.info("Size rule: ≤%.1fm → %.0fm   ≥%.1fm → %.0fm  (shape=%.2f)",
                     depth_shallow, elem_end_size, depth_deep, elem_start_size, shape)

        size_fn = make_size_fn(elem_end_size, elem_start_size, land_h,
                               depth_shallow, depth_deep, shape, factors[step])
        areas, h = triangle_areas(nodes, tris, depth_at, size_fn, grade)

        log.info("Step %d/%d  target edge %.1f / %.1f / %.1f m",
                 step+1, n_steps, h.min(), np.median(h), h.max())

        write_area_file(prefix, areas)
        run_triangle(f"-rp{q}a", prefix)
        report_quality(f"{file_name}.{step + 3}")

    # -------- Final quality-only passes (very important) --------
    last = num_iterrations
    for extra in range(1, 4):                     # 3 extra pure quality passes
        current = f"{file_name}.{last}"
        log.info("Final quality pass %d ...", extra)
        run_triangle(f"-rp{q}", current)          # no new area constraint
        last += 1
        report_quality(f"{file_name}.{last}")

    if plot:
        for k in range(1, last + 1):
            p = f"{file_name}.{k}"
            if Path(f"{p}.node").exists():
                plot_mesh(p, f"ITERATION_{k}.png")

    log.info("Finished. Best mesh is usually the highest number.")
    log.info("Check the quality reports above – look at Min angle and %% < 45°.")


if __name__ == "__main__":
    iter_mesh()
