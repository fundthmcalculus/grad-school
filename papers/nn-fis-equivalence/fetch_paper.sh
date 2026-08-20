#!/usr/bin/env bash
# Download the two Bede/Kreinovich/Toth equivalence papers into this directory.
#
# Run from a machine with ordinary network access. The session that created this
# directory could not: its egress proxy answers 403 to CONNECT for every
# scholarly host (see README.md), so nothing here was fetched.
#
#   ./fetch_paper.sh
#
# IJCCC is open access and should succeed unattended. The Springer chapter is
# paywalled; the script tries the DOI and reports what it got rather than
# leaving a 200-byte HTML error page named like a PDF.
set -uo pipefail
cd "$(dirname "$0")"

fetch() {
  local name="$1" url="$2"
  echo "==> $name"
  if curl -fsSL --max-time 120 -o "$name.pdf" "$url"; then
    if [ "$(head -c 4 "$name.pdf")" = "%PDF" ]; then
      echo "    ok: $name.pdf ($(wc -c <"$name.pdf") bytes)"
      return 0
    fi
    echo "    got a non-PDF response (paywall or interstitial); removing"
    rm -f "$name.pdf"
  else
    echo "    request failed"
  fi
  return 1
}

fetch "bede-kreinovich-toth-2025-ijccc-nd" \
  "https://univagora.ro/jour/index.php/ijccc/article/download/7127/pdf" ||
  echo "    try the landing page: https://univagora.ro/jour/index.php/ijccc/article/view/7127"

fetch "bede-kreinovich-toth-2023-nafips-1d" \
  "https://link.springer.com/content/pdf/10.1007/978-3-031-46778-3_5.pdf" ||
  cat <<'EOF'
    The NAFIPS 2023 chapter is behind Springer's paywall. Three routes:
      1. Institutional access via https://doi.org/10.1007/978-3-031-46778-3_5
      2. Kreinovich mirrors his work as UTEP CS technical reports --
         search https://scholarworks.utep.edu/cs_techrep/ for "ReLU"
      3. Ask him. He is on the committee.
EOF
