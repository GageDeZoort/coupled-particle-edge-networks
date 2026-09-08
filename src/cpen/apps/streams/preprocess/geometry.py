"""Sky geometry helpers for DES cell assignment and tangent-plane rotation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame


def radec_to_xyz(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float32))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float32))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def angular_distance_rad(xyz1, xyz2):
    xyz2 = np.atleast_2d(xyz2)
    dot = np.clip(np.sum(xyz1 * xyz2, axis=1), -1.0, 1.0)
    return np.arccos(dot)


def rotate_to_patch_frame(ra_deg, dec_deg, pmra, pmdec, ra0_deg, dec0_deg):
    """Rotate (ra, dec, pm) to local tangent plane at (ra0, dec0)."""
    c = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        pm_ra_cosdec=np.asarray(pmra, dtype=float) * u.mas / u.yr,
        pm_dec=np.asarray(pmdec, dtype=float) * u.mas / u.yr,
        frame="icrs",
    )
    center = SkyCoord(ra=float(ra0_deg) * u.deg, dec=float(dec0_deg) * u.deg, frame="icrs")
    c_off = c.transform_to(SkyOffsetFrame(origin=center))
    return (
        c_off.lon.to_value(u.deg),
        c_off.lat.to_value(u.deg),
        c_off.pm_lon_coslat.to_value(u.mas / u.yr),
        c_off.pm_lat.to_value(u.mas / u.yr),
    )


def rotate_positions_to_patch_frame(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Rotate sky positions to the tangent plane at ``(ra0, dec0)`` → ``(φ, λ)`` in deg."""
    c = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )
    center = SkyCoord(ra=float(ra0_deg) * u.deg, dec=float(dec0_deg) * u.deg, frame="icrs")
    c_off = c.transform_to(SkyOffsetFrame(origin=center))
    return c_off.lon.to_value(u.deg), c_off.lat.to_value(u.deg)


def patch_frame_to_radec(phi_deg, lam_deg, ra0_deg, dec0_deg):
    """Inverse of ``rotate_positions_to_patch_frame``."""
    center = SkyCoord(ra=float(ra0_deg) * u.deg, dec=float(dec0_deg) * u.deg, frame="icrs")
    off = SkyCoord(
        lon=np.asarray(phi_deg, dtype=float) * u.deg,
        lat=np.asarray(lam_deg, dtype=float) * u.deg,
        frame=SkyOffsetFrame(origin=center),
    )
    icrs = off.transform_to("icrs")
    return icrs.ra.degree, icrs.dec.degree


def circular_mean_ra_deg(ra_deg):
    """Circular mean of RA in degrees, in ``[0, 360)``."""
    r = np.deg2rad(np.asarray(ra_deg, dtype=float))
    ang = np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))
    return float(np.rad2deg(ang) % 360.0)


def assign_cells(df, centers, circle_radius_deg, ra_col="ra_des", dec_col="dec_des", verbose=True):
    """Assign each star to cells (overlapping circles). Adds in_cell_* bool columns."""
    ra = df[ra_col].to_numpy(np.float32)
    dec = df[dec_col].to_numpy(np.float32)
    m = np.isfinite(ra) & np.isfinite(dec)
    xyz = radec_to_xyz(ra[m], dec[m])
    r_rad = np.deg2rad(float(circle_radius_deg))

    n_centers = len(centers)
    centers_xyz = radec_to_xyz(
        np.array([c[0] for c in centers], dtype=np.float32),
        np.array([c[1] for c in centers], dtype=np.float32),
    )

    cell_cols = {}
    for j in range(n_centers):
        dist = angular_distance_rad(xyz, centers_xyz[j : j + 1])
        in_cell = np.zeros(len(df), dtype=bool)
        in_cell[m] = dist <= r_rad
        cell_cols[f"in_cell_{j + 1}"] = in_cell
    df = pd.concat([df, pd.DataFrame(cell_cols, index=df.index)], axis=1)

    if verbose:
        n_in_any = df[[f"in_cell_{k+1}" for k in range(n_centers)]].any(axis=1).sum()
        print("\n" + "=" * 72)
        print("Cell Assignment")
        print("=" * 72)
        print(f"Centers            : {n_centers}")
        print(f"Circle radius      : {circle_radius_deg} deg")
        print(f"Stars in ≥1 cell   : {n_in_any:,}")
        print(f"Stars in 0 cells   : {len(df) - n_in_any:,}\n")

    return df, n_centers
