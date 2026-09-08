#!/bin/bash
# Publishes the evaluation-layer dashboard. NO LLM in this loop — scan, render, deploy.
# Deploys the PROJECT dir (not a bare dir) so the edge auth function ships every time; deploying a
# plain dir would silently remove the site password on the next tick.
set -uo pipefail
REPO=/Users/dhivyavijayakumar/git/Percept-sdk
SITE=6b6e2ec3-7ad4-4020-9aa7-0662d70dff10   # percept-eval-layers — NOT the soak site
PROJ="$REPO/eval/dashboard/site_proj"
export PATH="/Users/dhivyavijayakumar/.nvm/versions/node/v22.13.0/bin:/usr/local/bin:/usr/bin:/bin"
cd "$REPO" || exit 1
{
  echo "--- $(date '+%F %T') ---"
  /usr/bin/python3 eval/dashboard/publish.py >/dev/null || { echo "render FAILED"; exit 4; }
  cp eval/dashboard/site/index.html "$PROJ/public/index.html"
  [ -f "$PROJ/netlify/edge-functions/auth.js" ] || { echo "REFUSING: edge auth missing"; exit 5; }
  (cd "$PROJ" && netlify deploy --prod --dir public --site "$SITE" 2>&1 | grep -E "Production URL|error" )
} >> eval/dashboard/logs/publish.log 2>&1
