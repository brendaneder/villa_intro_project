
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pact_views.py

Load NPZ outputs from:
  - Analytical solver (no EIR/SIR)
  - EIR-only pipeline
  - SIR+EIR pipeline

and generate:
  (1) A 1x3 panel of XY-plane images (top-down, "looking into Z")
  (2) A 1x3 panel of XZ-plane images (side view, "looking into Y")

USAGE (example):
  python plot_pact_views.py --analytic a_output.npz --eir eir_output.npz --sir sir_output.npz \
      --z_mm -10.0 --y_mm 0.0

If you don't pass arguments, it will try these defaults in the current directory:
  analytic: output_analytic.npz or output_data.npz
  eir:      output_eir.npz
  sir:      output_sir_eir.npz

Notes:
- For XY views, the script prefers a true volume key like one of:
    ["volume", "img_xyz", "das_cube", "recon_cube", "img_cube"]
  Shape should be (Nx, Ny, Nz). If unavailable, it will attempt to build a coarse XY slice by
  binning detector (x,y) positions and using a depth slice from an XZ image or time-of-flight proxy.
  If Ny == 1, the XY panel will be a "thin" single-row image (still valid).
- For XZ views, the script looks for any of these keys:
    ["das_img", "xz_img", "img_xz", "beamformed_xz"]
  If none found, it computes an XZ DAS from available traces and metadata.

Author: ChatGPT
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple, Optional, List

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -----------------------------
# Utilities
# -----------------------------
def _first_key(d: Dict, keys: List[str]):
    for k in keys:
        if k in d:
            return k
    return None


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as D:
        return {k: D[k] for k in D.files}


def _guess_file(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _compute_das_xz(pressure_traces: np.ndarray,
                    detectors: np.ndarray,
                    c: float,
                    t_min: float,
                    dt: float,
                    z_vals: np.ndarray,
                    y_mm: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an XZ image at a fixed y=y_mm by delay-and-sum from time traces.

    detectors: (Nd, 3) xyz in mm
    pressure_traces: (Nd, Nt)
    z_vals: array of z (depth) positions in mm, negative above origin typical
    Returns: (img_xz [Nx, Nz], x_coords [Nx], z_vals [Nz])
    """
    assert detectors.shape[1] == 3
    Nd, Nt = pressure_traces.shape

    # Use detector unique x positions as lateral X grid (sorted)
    x_coords = np.sort(np.unique(detectors[:, 0])).astype(np.float32)
    Nx = x_coords.size
    Nz = int(z_vals.size)

    # Build pixel grid in XZ at fixed y
    Xpix, Zpix = np.meshgrid(x_coords, z_vals, indexing='ij')  # (Nx, Nz)
    Npix = Nx * Nz
    pix = np.column_stack((Xpix.ravel(), np.full(Npix, y_mm, np.float32), Zpix.ravel())).astype(np.float32)

    # Distances from each pixel to each detector
    diff = pix[:, None, :] - detectors[None, :, :]         # (Npix, Nd, 3)
    dist = np.linalg.norm(diff, axis=2).astype(np.float32) # (Npix, Nd)

    # Fractional sample index per (pixel, detector)
    f = (dist / c - t_min) / dt
    f = np.clip(f, 0.0, Nt - 1.000001).astype(np.float32)
    i0 = np.floor(f).astype(np.int32)
    i1 = i0 + 1
    w = (f - i0).astype(np.float32)

    det_idx = np.broadcast_to(np.arange(Nd, dtype=np.int32)[None, :], (Npix, Nd))
    P0 = pressure_traces[det_idx, i0]
    P1 = pressure_traces[det_idx, i1]
    vals = (1.0 - w) * P0 + w * P1

    das_flat = vals.sum(axis=1, dtype=np.float32)   # (Npix,)
    img_xz = das_flat.reshape(Nx, Nz)

    return img_xz, x_coords, z_vals


def _two_slope_norm(arr: np.ndarray):
    return mcolors.TwoSlopeNorm(vmin=float(arr.min()), vcenter=0.0, vmax=float(arr.max()))


def _ensure_units(d: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    """
    Try to recover c [mm/us], t_min [us], dt [us] from the dict.
    """
    c = float(d["c"]) if "c" in d else 1.5  # mm/us
    t_vals = d.get("t_vals", None)
    if t_vals is not None:
        t_vals = np.asarray(t_vals).astype(np.float32)
        t_min = float(t_vals.min())
        if "dt" in d:
            dt = float(d["dt"])
        else:
            if t_vals.size > 1:
                dt = float(np.median(np.diff(t_vals)))
            else:
                dt = 1.0
    else:
        # last resort
        t_min, dt = 0.0, float(d.get("dt", 1.0))
    return c, t_min, dt


def _get_das_xz_from_dict(d: Dict[str, np.ndarray],
                          y_mm: float,
                          prefer_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull an XZ image out of a dictionary or recompute it.
    Returns: (img_xz, x_vals, z_vals)
    """
    # Attempt to find an existing XZ image
    if not prefer_recompute:
        key = _first_key(d, ["das_img", "xz_img", "img_xz", "beamformed_xz"])
        if key is not None:
            img = np.asarray(d[key])
            # Also try to find axes:
            xv = d.get("das_x_values", None)
            zv = d.get("das_z_values", None)
            if xv is None and "detectors" in d:
                xv = np.sort(np.unique(d["detectors"][:, 0].astype(np.float32)))
            if zv is None and "t_vals" in d and "c" in d:
                # infer z from time grid
                t_vals = d["t_vals"].astype(np.float32)
                zv = (d["c"].astype(np.float32) * t_vals).astype(np.float32) if np.isscalar(d["c"]) else float(d["c"]) * t_vals
            xv = np.asarray(xv) if xv is not None else np.arange(img.shape[0], dtype=np.float32)
            zv = np.asarray(zv) if zv is not None else np.arange(img.shape[1], dtype=np.float32)
            return img, xv.astype(np.float32), zv.astype(np.float32)

    # Otherwise recompute from traces
    required = ["detectors", "t_vals"]
    if not all(k in d for k in required):
        raise KeyError(f"Cannot build XZ image: missing one of {required}")
    traces_key = _first_key(d, ["data_eir_final", "pressure_composit", "pressure_analytic", "p_eir_only", "p_sir_eir"])
    if traces_key is None:
        raise KeyError("No suitable pressure trace array found to compute DAS (looked for data_eir_final, pressure_composit, pressure_analytic, p_eir_only, p_sir_eir).")
    traces = np.asarray(d[traces_key])
    detectors = np.asarray(d["detectors"]).astype(np.float32)
    c, t_min, dt = _ensure_units(d)
    t_vals = np.asarray(d["t_vals"]).astype(np.float32)
    # Choose reasonable z-range from t grid (z = c * t)
    z_vals = (c * t_vals).astype(np.float32)
    img_xz, xv, zv = _compute_das_xz(traces, detectors, c, t_min, dt, z_vals, y_mm=y_mm)
    return img_xz, xv, zv


def _get_xy_from_cube(d: Dict[str, np.ndarray], z_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to extract an XY slice at specified z from a 3D cube (Nx, Ny, Nz).
    Returns (img_xy, x_vals, y_vals)
    Raises KeyError if no cube-like data found.
    """
    cube_key = _first_key(d, ["volume", "img_xyz", "das_cube", "recon_cube", "img_cube"])
    if cube_key is None:
        raise KeyError("No (Nx, Ny, Nz) cube found for XY slicing.")
    cube = np.asarray(d[cube_key])
    if cube.ndim != 3:
        raise ValueError(f"Cube key '{cube_key}' must be 3D (Nx,Ny,Nz); got shape {cube.shape}")
    Nx, Ny, Nz = cube.shape
    # Try to get axes
    x_vals = d.get("x_vals", None)
    y_vals = d.get("y_vals", None)
    z_vals = d.get("z_vals", None)
    if x_vals is None and "detectors" in d:
        x_vals = np.sort(np.unique(d["detectors"][:, 0].astype(np.float32)))
        if len(x_vals) != Nx:
            x_vals = np.arange(Nx, dtype=np.float32)
    if y_vals is None and "detectors" in d:
        y_vals = np.sort(np.unique(d["detectors"][:, 1].astype(np.float32)))
        if len(y_vals) != Ny:
            y_vals = np.arange(Ny, dtype=np.float32)
    if z_vals is None and "t_vals" in d and "c" in d:
        t_vals = np.asarray(d["t_vals"]).astype(np.float32)
        c = float(d["c"]) if "c" in d else 1.5
        z_vals = (c * t_vals).astype(np.float32)
        if len(z_vals) != Nz:
            z_vals = np.arange(Nz, dtype=np.float32)
    if any(v is None for v in (x_vals, y_vals, z_vals)):
        # Fallback
        x_vals = np.arange(Nx, dtype=np.float32) if x_vals is None else x_vals
        y_vals = np.arange(Ny, dtype=np.float32) if y_vals is None else y_vals
        z_vals = np.arange(Nz, dtype=np.float32) if z_vals is None else z_vals

    # Find nearest z index
    z_idx = int(np.argmin(np.abs(np.asarray(z_vals) - z_mm)))
    img_xy = cube[:, :, z_idx]
    return img_xy, np.asarray(x_vals, dtype=np.float32), np.asarray(y_vals, dtype=np.float32)


def _best_effort_xy_from_xz(img_xz: np.ndarray, xv: np.ndarray, zv: np.ndarray, z_mm: float,
                            detector_y_vals: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If no 3D cube exists, build a thin XY image by selecting the XZ slice at depth z_mm,
    and tiling across available detector y positions (or a single row if Ny=1).
    """
    z_idx = int(np.argmin(np.abs(zv - z_mm)))
    x_line = img_xz[:, z_idx]  # (Nx,)
    if detector_y_vals is not None and detector_y_vals.size > 0:
        y_vals = np.sort(np.unique(detector_y_vals.astype(np.float32)))
    else:
        y_vals = np.array([0.0], dtype=np.float32)
    Ny = y_vals.size

    # Tile the X line across Y to make an (Nx, Ny) image
    img_xy = np.repeat(x_line[:, None], Ny, axis=1)
    return img_xy, xv.astype(np.float32), y_vals


# -----------------------------
# Plotting helpers
# -----------------------------
def _imshow(ax, img, extent, title: str):
    im = ax.imshow(img.T, extent=extent, aspect='auto', origin='upper',
                   cmap='seismic', norm=_two_slope_norm(img))
    ax.set_title(title)
    return im


def main():
    ap = argparse.ArgumentParser(description="Render XY and XZ views from PACT NPZ outputs (Analytic, EIR, SIR+EIR).")
    ap.add_argument("--analytic", type=str, default=None, help="NPZ from analytical solver (no EIR/SIR).")
    ap.add_argument("--eir",      type=str, default=None, help="NPZ from EIR-only pipeline.")
    ap.add_argument("--sir",      type=str, default=None, help="NPZ from SIR+EIR pipeline.")
    ap.add_argument("--z_mm",     type=float, default=-10.0, help="Depth (z, mm) for XY slice (looking down).")
    ap.add_argument("--y_mm",     type=float, default=0.0,   help="Y (mm) for XZ slice (side view).")
    ap.add_argument("--prefer_recompute_xz", action="store_true",
                    help="Ignore any stored XZ images and recompute DAS from traces instead.")
    args = ap.parse_args()

    # Guess defaults if not provided
    analytic_path = args.analytic or _guess_file(["output_analytic.npz", "output_data.npz"])
    eir_path      = args.eir      or _guess_file(["output_eir.npz"])
    sir_path      = args.sir      or _guess_file(["output_sir_eir.npz", "output_sir.npz"])

    missing = [name for name, p in [("analytic", analytic_path), ("eir", eir_path), ("sir", sir_path)] if p is None]
    if missing:
        print("Could not locate NPZ files for:", ", ".join(missing))
        print("Pass them explicitly via --analytic, --eir, --sir")
        sys.exit(1)

    D_ana = _load_npz(analytic_path)
    D_eir = _load_npz(eir_path)
    D_sir = _load_npz(sir_path)

    # ---- Build/Load XZ images (one per method) ----
    try:
        xz_ana, xv, zv = _get_das_xz_from_dict(D_ana, args.y_mm, prefer_recompute=args.prefer_recompute_xz)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain XZ for ANALYTIC: {e}")
    try:
        xz_eir, _, _ = _get_das_xz_from_dict(D_eir, args.y_mm, prefer_recompute=args.prefer_recompute_xz)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain XZ for EIR: {e}")
    try:
        xz_sir, _, _ = _get_das_xz_from_dict(D_sir, args.y_mm, prefer_recompute=args.prefer_recompute_xz)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain XZ for SIR+EIR: {e}")

    # ---- Build/Load XY images (one per method) ----
    # Prefer true cube; else best-effort from XZ selection at z_mm
    det_y_ana = D_ana.get("detectors", None)
    det_y_vals = det_y_ana[:, 1] if isinstance(det_y_ana, np.ndarray) and det_y_ana.shape[1] >= 2 else None

    try:
        xy_ana, xv_xy, yv_xy = _get_xy_from_cube(D_ana, args.z_mm)
    except Exception:
        xy_ana, xv_xy, yv_xy = _best_effort_xy_from_xz(xz_ana, xv, zv, args.z_mm, det_y_vals)

    try:
        xy_eir, _, _ = _get_xy_from_cube(D_eir, args.z_mm)
    except Exception:
        xy_eir, _, _ = _best_effort_xy_from_xz(xz_eir, xv, zv, args.z_mm, det_y_vals)

    try:
        xy_sir, _, _ = _get_xy_from_cube(D_sir, args.z_mm)
    except Exception:
        xy_sir, _, _ = _best_effort_xy_from_xz(xz_sir, xv, zv, args.z_mm, det_y_vals)

    # ---- Figure 1: XY panel (3 subplots) ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent_xy = [float(xv_xy.min()), float(xv_xy.max()), float(yv_xy.min()), float(yv_xy.max())]
    im1 = _imshow(axes1[0], xy_ana, extent_xy, "Analytical (XY @ z = {:.2f} mm)".format(args.z_mm))
    im2 = _imshow(axes1[1], xy_eir, extent_xy, "EIR only (XY)")
    im3 = _imshow(axes1[2], xy_sir, extent_xy, "SIR + EIR (XY)")
    axes1[0].set_xlabel("x (mm)"); axes1[0].set_ylabel("y (mm)")
    axes1[1].set_xlabel("x (mm)"); axes1[1].set_ylabel("y (mm)")
    axes1[2].set_xlabel("x (mm)"); axes1[2].set_ylabel("y (mm)")
    # One shared colorbar
    cbar1 = fig1.colorbar(im3, ax=axes1.ravel().tolist(), shrink=0.85, label="Amplitude (arb.)")
    fig1.suptitle("Top-down XY plane (looking into Z)", fontsize=13)

    # ---- Figure 2: XZ panel (3 subplots) ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent_xz = [float(xv.min()), float(xv.max()), float(zv.min()), float(zv.max())]
    im4 = _imshow(axes2[0], xz_ana, extent_xz, "Analytical (XZ @ y = {:.2f} mm)".format(args.y_mm))
    im5 = _imshow(axes2[1], xz_eir, extent_xz, "EIR only (XZ)")
    im6 = _imshow(axes2[2], xz_sir, extent_xz, "SIR + EIR (XZ)")
    for ax in axes2:
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    cbar2 = fig2.colorbar(im6, ax=axes2.ravel().tolist(), shrink=0.85, label="Amplitude (arb.)")
    fig2.suptitle("Side view XZ plane (looking into Y)", fontsize=13)

    plt.show()


if __name__ == "__main__":
    main()
