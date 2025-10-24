#!/bin/bash
set -euo pipefail
# Source Spack for the main process
[ -f "$HOME/spack/share/spack/setup-env.sh" ] && . "$HOME/spack/share/spack/setup-env.sh"
exec "$@"