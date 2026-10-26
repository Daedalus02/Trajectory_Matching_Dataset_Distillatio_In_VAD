#!/usr/bin/env bash
# Builds continual/moviad: MoViAD at the commit used in the thesis + the thesis changes
# (3 modified files, 2 new replay strategies, the experiment scripts). Needs git and uv.
# Usage: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

[ -d moviad ] || git clone https://github.com/AMCO-UniPD/moviad.git moviad
git -C moviad checkout ed856ec7cb087f18e86c3889913414b637357c05

# overlay: modified + new files (same content as moviad_thesis.patch for the modified ones)
cp -r overlay/. moviad/
cp experiments/*.py moviad/experiments/

cd moviad && uv sync
echo "MoViAD ready in continual/moviad"
