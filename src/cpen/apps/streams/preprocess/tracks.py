"""Minimal on-sky track fits for mock streams (PCA line in a tangent plane)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
from cpen.apps.streams.preprocess.geometry import (
    circular_mean_ra_deg,
    patch_frame_to_radec,
    rotate_positions_to_patch_frame,
)


@dataclass
class LinearSkyTrack:
    """Straight track: ``(φ, λ) = μ + s · direction`` in the stream tangent plane."""

    stream_label: str
    ra0: float
    dec0: float
    mu: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    s_min: float
    s_max: float
    rms_perp: float
    n_mock: int


def fit_linear_sky_track(
    ra,
    dec,
    *,
    stream_label="",
    pad_frac=0.08,
    min_stars=10,
) -> LinearSkyTrack:
    """PCA line through mock members in the tangent plane at their centroid."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[ok], dec[ok]
    if len(ra) < min_stars:
        raise ValueError(f"Need ≥{min_stars} stars to fit a track, got {len(ra)}")

    ra0 = circular_mean_ra_deg(ra)
    dec0 = float(np.median(dec))
    phi, lam = rotate_positions_to_patch_frame(ra, dec, ra0, dec0)
    xy = np.column_stack([phi, lam])
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    s = (xy - mu) @ direction
    d = (xy - mu) @ normal
    span = float(s.max() - s.min())
    pad = pad_frac * span if span > 0 else 0.1
    return LinearSkyTrack(
        stream_label=str(stream_label),
        ra0=ra0,
        dec0=dec0,
        mu=mu,
        direction=direction,
        normal=normal,
        s_min=float(s.min() - pad),
        s_max=float(s.max() + pad),
        rms_perp=float(np.sqrt(np.mean(d * d))),
        n_mock=int(len(ra)),
    )


def project_on_track(ra, dec, track: LinearSkyTrack):
    """Return along-track ``s`` and perpendicular ``d`` (both degrees)."""
    phi, lam = rotate_positions_to_patch_frame(ra, dec, track.ra0, track.dec0)
    xy = np.column_stack([phi, lam]) - track.mu
    s = xy @ track.direction
    d = xy @ track.normal
    return s, d


def track_curve_radec(track: LinearSkyTrack, n=80):
    """RA/Dec samples along the fitted segment."""
    s = np.linspace(track.s_min, track.s_max, n)
    xy = track.mu + s[:, None] * track.direction
    return patch_frame_to_radec(xy[:, 0], xy[:, 1], track.ra0, track.dec0)


def track_tube_radec(track: LinearSkyTrack, half_width, n=80):
    """RA/Dec of the two tube edges (parallel to the track, ±half_width)."""
    s = np.linspace(track.s_min, track.s_max, n)
    xy = track.mu + s[:, None] * track.direction
    off = half_width * track.normal
    ra_lo, dec_lo = patch_frame_to_radec(xy[:, 0] - off[0], xy[:, 1] - off[1], track.ra0, track.dec0)
    ra_hi, dec_hi = patch_frame_to_radec(xy[:, 0] + off[0], xy[:, 1] + off[1], track.ra0, track.dec0)
    return (ra_lo, dec_lo), (ra_hi, dec_hi)


def _rough_sky_box(ra, dec, ra0, dec0, radius_deg):
    dra = ((np.asarray(ra, dtype=float) - ra0 + 180.0) % 360.0) - 180.0
    ddec = np.asarray(dec, dtype=float) - dec0
    cosd = np.cos(np.deg2rad(dec0))
    return (dra * cosd) ** 2 + ddec ** 2 <= radius_deg ** 2


def fit_all_mock_tracks(
    df,
    *,
    ra_col="ra_des",
    dec_col="dec_des",
    min_stars=30,
):
    """Fit a linear sky track for every mock stream with ≥ ``min_stars`` members."""
    df = ensure_is_mock_stream(df)
    mock = df.loc[df["is_mock_stream"].to_numpy(bool)]
    tracks = {}
    for label, part in mock.groupby("stream_label"):
        ra = part[ra_col].to_numpy()
        dec = part[dec_col].to_numpy()
        ok = np.isfinite(ra) & np.isfinite(dec)
        if int(ok.sum()) < min_stars:
            continue
        tracks[str(label)] = fit_linear_sky_track(
            ra[ok], dec[ok], stream_label=str(label), min_stars=min_stars,
        )
    return tracks


def mock_track_summary(tracks: dict[str, LinearSkyTrack]) -> pd.DataFrame:
    rows = [
        {
            "stream_label": t.stream_label,
            "n_mock": t.n_mock,
            "length_deg": t.s_max - t.s_min,
            "rms_perp_deg": t.rms_perp,
            "ra0": t.ra0,
            "dec0": t.dec0,
        }
        for t in tracks.values()
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_mock", ascending=False).reset_index(drop=True)
