#!/usr/bin/env python3
"""
pyTri: Python driver for Triangle (https://www.cs.cmu.edu/~quake/triangle.html)

Generates triangular meshes for geophysical domains, refined iteratively
using bathymetry, for hydrodynamic models such as Swash, Swan and Thetis.

Original author: Shuaib Rasheed (30/08/2021)

Modified for large bathymetry files (RegularGridInterpolator + binning) (04/10/2026).
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
EQUILATERAL = np.sqrt(3.0) / 4.0  # area = EQUILATERAL * edge**2


def run_triangle(switches, target):
    """Run Triangle (no shell, fails loudly)."""
    cmd = ["triangle", switches, str(target)]
    log.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        log.error("Triangle failed (exit %s):\n%s\n%s", exc.returncode, exc.stdout, exc.stderr)
        raise
    log.debug(result.stdout)


def load_mesh(prefix):
    """Read Triangle .node/.ele. Returns (nodes[n,2], triangles[m,3]) 0-based."""
    node_data = np.loadtxt(f"{prefix}.node", skiprows=1, usecols=(0, 1, 2), ndmin=2)
    tris = np.loadtxt(f"{prefix}.ele", skiprows=1, usecols=(1, 2, 3), dtype=int, ndmin=2)
    base = int(node_data[0, 0])
    return node_data[:, 1:3], tris - base


def mean_edge_lengths(nodes, tris):
    """Mean side length of every triangle."""
    p = nodes[tris]
    d01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    d12 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    d20 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    return (d01 + d12 + d20) / 3.0


def write_area_file(prefix, areas):
    """Write a Triangle .area file (max area per triangle)."""
    index = np.arange(1, len(areas) + 1)
    with open(f"{prefix}.area", "w") as f:
        f.write(f"{len(areas)}\n")
        np.savetxt(f, np.column_stack([index, areas]), fmt=["%d", "%.6f"])


def make_bathy_interpolator(bathy_file, resolution=50.0, elevation=False):
    """
    Load a large x,y,bathy CSV, bin it onto a regular grid, and return
    a fast RegularGridInterpolator + a depth_at helper.
    """
    log.info("Loading bathymetry from %s ...", bathy_file)
    df = pd.read_csv(bathy_file, header=0)          # skips the header automatically
    # Expect columns: x, y, bathy  (adjust iloc if your order is different)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    z = df.iloc[:, 2].to_numpy(dtype=float)
    log.info("  %d points loaded", len(x))

    # --- bin onto a regular grid ------------------------------------------
    dx = dy = float(resolution)
    x_edges = np.arange(x.min(), x.max() + dx, dx)
    y_edges = np.arange(y.min(), y.max() + dy, dy)

    log.info("  Binning onto %.1f m grid (%d × %d cells) ...",
             resolution, len(x_edges)-1, len(y_edges)-1)

    Z, _, _, _ = binned_statistic_2d(
        x, y, z, statistic="mean", bins=[x_edges, y_edges]
    )

    # fill empty bins with nearest neighbour
    mask = np.isnan(Z)
    if mask.any():
        ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
        Z = Z[tuple(ind)]

    # cell centres
    X = (x_edges[:-1] + x_edges[1:]) * 0.5
    Y = (y_edges[:-1] + y_edges[1:]) * 0.5

    # RegularGridInterpolator expects (y, x) order for the axes
    interp = RegularGridInterpolator(
        (Y, X), Z.T, bounds_error=False, fill_value=None
    )

    def depth_at(points):
        """Return positive depth at the given (N,2) points."""
        # points are (x,y) → we need (y,x) for the interpolator
        d = interp(points[:, ::-1])
        # convert elevation → depth if necessary
        return -d if elevation else d

    # quick statistics
    sample = depth_at(np.column_stack([x[::max(1, len(x)//10000)],
                                       y[::max(1, len(y)//10000)]]))
    wet = sample > 0
    log.info("  Bathymetry ready: %.0f%% water, %.0f%% land (sample)",
             100 * wet.mean(), 100 * (1 - wet.mean()))
    return depth_at


def limit_gradient(nodes, tris, h, grade, max_iter=200):
    """Enforce |h_i - h_j| <= grade * dist(i, j) along mesh edges (only ever lowers h)."""
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


def make_size_fn(mode, h_min, h_max, land_h, alpha=None, d_shallow=None, d_deep=None,
                 shape=1.0, factor=1.0):
    """Return f(depth) -> target edge length. depth <= 0 is land and gets land_h."""
    def size_fn(depth):
        with np.errstate(invalid="ignore"):
            if mode == "ratio":
                h = depth / alpha
            else:
                s = np.clip((depth - d_shallow) / (d_deep - d_shallow), 0.0, 1.0) ** shape
                h = h_min + (h_max - h_min) * s
            h = np.clip(h * factor, h_min, h_max)
        return np.where(depth <= 0, land_h, h)
    return size_fn


def triangle_areas(nodes, tris, depth_at, size_fn, grade):
    """Max area per triangle from bathymetry. Returns (areas, edge_lengths)."""
    h_node = size_fn(depth_at(nodes))
    h_node = limit_gradient(nodes, tris, h_node, grade)
    centroids = nodes[tris].mean(axis=1)
    h_cent = size_fn(depth_at(centroids))
    h_tri = np.minimum(h_node[tris].min(axis=1), h_cent)
    return EQUILATERAL * h_tri**2, h_tri


def plot_mesh(prefix, out_png):
    """Mesh coloured by triangle size."""
    nodes, tris = load_mesh(prefix)
    fig, ax = plt.subplots(figsize=(10, 9))
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    size = mean_edge_lengths(nodes, tris)
    pc = ax.tripcolor(tri, facecolors=size, cmap="viridis_r", edgecolors="k", linewidth=0.15)
    pc.set_clim(*np.percentile(size, [1, 99]))
    fig.colorbar(pc, ax=ax, shrink=0.8, label="mean triangle edge length (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{Path(prefix).name}: {len(tris)} triangles")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


@click.command()
@click.option("--file_name", default="TEST", help="Name of the .poly file (without extension).")
@click.option("--bathy_file_name", default="bath.csv",
              help="Bathymetry CSV with one header line. Columns: x,y,bathy")
@click.option("--bathy_res", default=50.0, type=float,
              help="Grid resolution (metres) used to bin the bathymetry. Smaller = more detail but slower.")
@click.option("--num_iterrations", default=5, type=int, help="Number of meshes to generate.")
@click.option("--elem_start_size", default=10000.0, type=float,
              help="Coarsest triangle EDGE LENGTH in m (deep water).")
@click.option("--elem_end_size", default=100.0, type=float,
              help="Finest triangle EDGE LENGTH in m (shallow water).")
@click.option("--size_mode", type=click.Choice(["range", "ratio"]), default="range",
              help="range: sizes follow depth between depth_shallow and depth_deep (default). "
                   "ratio: edge = depth/alpha (original).")
@click.option("--depth_shallow", default=None, type=float,
              help="[range] depth (m) at/above which the finest size is used. Default: 5th percentile.")
@click.option("--depth_deep", default=None, type=float,
              help="[range] depth (m) at/below which the coarsest size is used. Default: 95th percentile.")
@click.option("--shape", default=1.0, type=float,
              help="[range] curve exponent. 1 = linear; >1 keeps fine sizes over more of the "
                   "shallows; <1 coarsens sooner.")
@click.option("--init_ratio", default=None, type=float,
              help="[ratio] initial alpha. Default: x_scale/5000.")
@click.option("--final_ratio", default=None, type=float,
              help="[ratio] final alpha. Default: 10 x init_ratio.")
@click.option("--grade", default=0.3, type=float,
              help="Max rate of change of edge length per metre (smoothness). Lower = smoother.")
@click.option("--elevation", is_flag=True,
              help="Bathymetry column is elevation (negative below sea level) instead of depth.")
@click.option("--convex_hull", is_flag=True,
              help="Mesh the whole convex hull (Triangle -c). Leave OFF for non-convex/irregular boundaries.")
@click.option("--land", "land_mode", type=click.Choice(["coarse", "fine"]), default="coarse",
              help="Size at land points (depth <= 0). Default: coarse.")
@click.option("--x_scale", default=1000.0, type=float, help="[ratio] x length scale (m); sets default alpha.")
@click.option("--y_scale", default=1000.0, type=float, help="Unused; kept for compatibility.")
@click.option("--min_angle", default=20.0, type=float, help="Triangle -q minimum angle (max ~28.6).")
@click.option("--plot/--no-plot", default=True, help="Save a PNG of each mesh, coloured by size.")
@click.option("-v", "--verbose", is_flag=True, help="Show Triangle output.")
def iter_mesh(file_name, bathy_file_name, bathy_res, num_iterrations, elem_start_size, elem_end_size,
              size_mode, depth_shallow, depth_deep, shape, init_ratio, final_ratio, grade,
              elevation, convex_hull, land_mode, x_scale, y_scale, min_angle, plot, verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if shutil.which("triangle") is None:
        sys.exit("Triangle is not installed or not on PATH: "
                 "https://www.cs.cmu.edu/~quake/triangle.html")
    if not Path(f"{file_name}.poly").is_file():
        sys.exit(f"Cannot find {file_name}.poly")
    if elem_start_size <= elem_end_size:
        sys.exit("elem_start_size must be larger than elem_end_size.")
    if num_iterrations < 2:
        sys.exit("num_iterrations must be at least 2.")
    if depth_shallow is not None and depth_deep is not None and depth_deep <= depth_shallow:
        sys.exit("depth_deep must be larger than depth_shallow.")

    n_steps = num_iterrations - 2
    if size_mode == "ratio":
        if init_ratio is None:
            init_ratio = x_scale / 5000.0
        if final_ratio is None:
            final_ratio = init_ratio * 10.0
        if init_ratio >= final_ratio:
            sys.exit("init_ratio must be smaller than final_ratio.")
        alphas = np.array([final_ratio]) if n_steps == 1 else np.linspace(init_ratio, final_ratio, n_steps)
    factors = np.geomspace(3.0, 1.0, n_steps) if n_steps > 1 else np.array([1.0])

    # ---- fast bathymetry interpolator ------------------------------------
    depth_at = make_bathy_interpolator(bathy_file_name, resolution=bathy_res, elevation=elevation)

    land_h = elem_start_size if land_mode == "coarse" else elem_end_size

    # ---- initial meshes: coarse everywhere -------------------------------
    q = f"q{min_angle:g}e"
    run_triangle(f"-p{'c' if convex_hull else ''}{q}", f"{file_name}.poly")
    run_triangle(f"-rp{q}a{EQUILATERAL * elem_start_size**2:.6f}", f"{file_name}.1")

    # ---- bathymetry-driven refinement ------------------------------------
    for step in range(n_steps):
        prefix = f"{file_name}.{step + 2}"
        nodes, tris = load_mesh(prefix)

        if size_mode == "range" and (depth_shallow is None or depth_deep is None):
            d = depth_at(nodes[tris].mean(axis=1))
            d = d[d > 0]
            if len(d) == 0:
                sys.exit("No positive depths found in the domain. Check the bathymetry file, "
                         "--elevation, or set --depth_shallow / --depth_deep.")
            lo, hi = np.percentile(d, [5, 95])
            depth_shallow = lo if depth_shallow is None else depth_shallow
            depth_deep = hi if depth_deep is None else depth_deep
            if depth_deep <= depth_shallow:
                sys.exit(f"Depth range is degenerate ({depth_shallow:.2f}..{depth_deep:.2f} m); "
                         "set --depth_shallow and --depth_deep explicitly.")
        if size_mode == "range" and step == 0:
            log.info("Size rule: depth <= %.1f m -> %.0f m edges, depth >= %.1f m -> %.0f m edges (shape %.2g)",
                     depth_shallow, elem_end_size, depth_deep, elem_start_size, shape)

        size_fn = make_size_fn(size_mode, elem_end_size, elem_start_size, land_h,
                               alpha=alphas[step] if size_mode == "ratio" else None,
                               d_shallow=depth_shallow, d_deep=depth_deep, shape=shape,
                               factor=1.0 if size_mode == "ratio" else factors[step])
        areas, h = triangle_areas(nodes, tris, depth_at, size_fn, grade)
        log.info("Step %d/%d target edge min/median/max = %.1f / %.1f / %.1f m",
                 step + 1, n_steps, h.min(), np.median(h), h.max())
        if h.max() / h.min() < 1.5:
            log.warning("Target size is almost uniform. Check the depth range/sign (--elevation), "
                        "elem_start/end_size, and (ratio mode) alpha.")
        write_area_file(prefix, areas)
        run_triangle(f"-rp{q}a", prefix)

        n2, t2 = load_mesh(f"{file_name}.{step + 3}")
        e = mean_edge_lengths(n2, t2)
        log.info("   -> mesh %d: %d triangles, edge min/median/max = %.1f / %.1f / %.1f m",
                 step + 3, len(t2), e.min(), np.median(e), e.max())

    if plot:
        for k in range(1, num_iterrations + 1):
            log.info("Plotting mesh %d", k)
            plot_mesh(f"{file_name}.{k}", f"ITERATION_{k}.png")
    log.info("Done.")


if __name__ == "__main__":
    iter_mesh()
