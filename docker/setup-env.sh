#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="${SCRIPT_DIR}/.env"
EXAMPLE_FILE="${SCRIPT_DIR}/.env.example"

if [ -f "${ENV_FILE}" ]; then
    echo "${ENV_FILE} already exists"
    exit 0
fi

cp "${EXAMPLE_FILE}" "${ENV_FILE}"

sed -i "s/^USER=.*/USER=${USER}/" "${ENV_FILE}"
sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/" "${ENV_FILE}"
sed -i "s/^HOST_GID=.*/HOST_GID=$(id -g)/" "${ENV_FILE}"

if [ -n "${DISPLAY:-}" ]; then
    sed -i "s|^DISPLAY=.*|DISPLAY=${DISPLAY}|" "${ENV_FILE}"
fi

echo "Generated ${ENV_FILE}"