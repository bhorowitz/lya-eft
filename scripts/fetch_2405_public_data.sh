#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/external/cup1d/data/p1d_measurements"

mkdir -p "${DATA_DIR}/Chabanier2019" "${DATA_DIR}/eBOSS_mock"

base_chab="https://raw.githubusercontent.com/igmhub/cup1d/main/data/p1d_measurements/Chabanier2019"
base_mock="https://raw.githubusercontent.com/igmhub/cup1d/main/data/p1d_measurements/eBOSS_mock"

echo "Fetching Chabanier2019 DR14 1D data/covariance..."
curl -fsSL "${base_chab}/Pk1D_data.dat" -o "${DATA_DIR}/Chabanier2019/Pk1D_data.dat"
curl -fsSL "${base_chab}/Pk1D_cor.dat" -o "${DATA_DIR}/Chabanier2019/Pk1D_cor.dat"
curl -fsSL "${base_chab}/Pk1D_syst.dat" -o "${DATA_DIR}/Chabanier2019/Pk1D_syst.dat"
curl -fsSL "${base_chab}/README" -o "${DATA_DIR}/Chabanier2019/README"

echo "Fetching eBOSS mock files (used as LaCE-style synthetic calibration proxy)..."
curl -fsSL "${base_mock}/pk_1d_Nyx_emu_fiducial_mock.out" -o "${DATA_DIR}/eBOSS_mock/pk_1d_Nyx_emu_fiducial_mock.out"
curl -fsSL "${base_mock}/pk_1d_DR12_13bins_invCov.out" -o "${DATA_DIR}/eBOSS_mock/pk_1d_DR12_13bins_invCov.out"

echo "Done."
echo "Stored under: ${DATA_DIR}"
