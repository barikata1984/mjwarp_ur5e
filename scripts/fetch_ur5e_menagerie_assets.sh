#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TARGET_DIR="${REPO_ROOT}/assets/ur5e/mjcf"
TMP_DIR=$(mktemp -d)
REPO_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
SOURCE_DIR="${TMP_DIR}/mujoco_menagerie/universal_robots_ur5e"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TARGET_DIR}"

git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TMP_DIR}/mujoco_menagerie" >/dev/null 2>&1
git -C "${TMP_DIR}/mujoco_menagerie" sparse-checkout set universal_robots_ur5e >/dev/null 2>&1

rm -rf "${TARGET_DIR}/assets"
cp -R "${SOURCE_DIR}/assets" "${TARGET_DIR}/assets"
cp "${SOURCE_DIR}/scene.xml" "${TARGET_DIR}/scene.xml"
cp "${SOURCE_DIR}/ur5e.xml" "${TARGET_DIR}/ur5e.xml"
cp "${SOURCE_DIR}/LICENSE" "${TARGET_DIR}/LICENSE.menagerie"
cp "${SOURCE_DIR}/README.md" "${TARGET_DIR}/README.menagerie.md"
cp "${SOURCE_DIR}/CHANGELOG.md" "${TARGET_DIR}/CHANGELOG.menagerie.md"

echo "Synced UR5e MJCF assets into ${TARGET_DIR}"