SPLITS = ["train", "test", "val"]

TRAIN_VARS = [
    "ra_des_sin", "ra_des_cos", "dec_des",
    "psf_mag_aper_8_g_corrected_des", "psf_mag_aper_8_r_corrected_des",
    #"psf_mag_aper_8_b_corrected_des",
    "pmra_gaia", "pmdec_gaia", "parallax_gaia",
]

TRAJ_VARS = [
    "ra_des", "dec_des", "pmra_gaia", "pmdec_gaia"
]

IMPUTE_VARS = [
    "ra_des", "dec_des",
    "psf_mag_aper_8_g_corrected_des", "psf_mag_aper_8_r_corrected_des",
    #"psf_mag_aper_8_b_corrected_des",
    "pmra_gaia", "pmdec_gaia", "parallax_gaia",
]

TRUTH_VARS = [
    "stream_id", "is_mock", "galaxy_id", "split"
]

TEST_REAL_STREAMS = {"Elqui", "Chenab", "Indus", "Phoenix", "AAU"}
MERGE_TO_AAU = {"ATLAS", "Aliqa"}
BACKGROUND = "Background"

# Mock streams smaller than this are demoted to field at load (not a cell gate).
MIN_MOCK_STREAM_STARS = 100

# cut g/r band magnitudes above thresholds
PSF_G_UPPER = 22
PSF_R_UPPER = 20.5

# Physical cuts, frozen from train_galaxy_0000 (do not refit per galaxy).
# Linear CMD slab: r = m(g-r) + b, keep |orthogonal distance| ≤ half_width.
# m, b = PCA isochrone on 0000 mocks; half_width = first w with mock keep ≥ 90%.
PI_MAX = 1.0  # mas; keep π < PI_MAX (missing parallax kept)
CMD_SLAB_M = -14.303846
CMD_SLAB_B = 24.233482
CMD_SLAB_HALF_WIDTH = 0.201538
CMD_G_COL = "psf_mag_aper_8_g_corrected_des"
CMD_R_COL = "psf_mag_aper_8_r_corrected_des"
CMD_PARALLAX_COL = "parallax_gaia"

DEFAULT_GALAXY_ROOT = (
    "/projects/BHANIN/adri_gage_gnn_streams/gnn_streams/mocks/"
    "galaxy_with_background"
)
DEFAULT_PIPELINE_ROOT = "/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies"
DEFAULT_REAL_PIPELINE_ROOT = "/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies_real"
# Tuning galaxy: skip train/0000 by default (cuts were chosen there).
TUNE_SPLIT = "train"
TUNE_GALAXY_ID = "0000"
