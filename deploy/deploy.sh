#!/usr/bin/env bash
# Zero-downtime blue/green deploy for the XR QA container.
#
# Usage: deploy.sh <IMAGE_URI>
# Example: deploy.sh 123456789.dkr.ecr.us-east-1.amazonaws.com/xr-qa:abc1234
#
# Strategy:
#   1. Pull the new image
#   2. Start a "green" container on port 8001
#   3. Health-check green (up to 60 s)
#   4. Swap traffic: update the nginx upstream to green
#   5. Stop the old "blue" container
set -euo pipefail

IMAGE_URI="${1:?Usage: deploy.sh <IMAGE_URI>}"
APP_PORT=8000
GREEN_PORT=8001
CONTAINER_BLUE="xr-qa-blue"
CONTAINER_GREEN="xr-qa-green"
NGINX_UPSTREAM="/etc/nginx/conf.d/xr-qa-upstream.conf"
HEALTH_URL="http://localhost:${GREEN_PORT}/health"
HEALTH_TIMEOUT=60

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ── 1. Pull ───────────────────────────────────────────────────────────────────
log "Pulling $IMAGE_URI …"
docker pull "$IMAGE_URI"

# ── 2. Start green ────────────────────────────────────────────────────────────
log "Starting green container on :${GREEN_PORT} …"
docker rm -f "$CONTAINER_GREEN" 2>/dev/null || true
docker run -d \
  --name "$CONTAINER_GREEN" \
  --restart unless-stopped \
  -p "${GREEN_PORT}:8000" \
  "$IMAGE_URI"

# ── 3. Health-check green ─────────────────────────────────────────────────────
log "Waiting for green to be healthy (timeout ${HEALTH_TIMEOUT}s) …"
elapsed=0
until curl -sf "$HEALTH_URL" >/dev/null 2>&1; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
    log "ERROR: green failed health check – rolling back."
    docker rm -f "$CONTAINER_GREEN"
    exit 1
  fi
done
log "Green is healthy."

# ── 4. Swap nginx upstream ────────────────────────────────────────────────────
if command -v nginx &>/dev/null; then
  log "Switching nginx upstream to :${GREEN_PORT} …"
  cat > "$NGINX_UPSTREAM" <<EOF
upstream xr_qa_backend {
    server 127.0.0.1:${GREEN_PORT};
}
EOF
  nginx -t && nginx -s reload
  log "Nginx reloaded."
else
  log "Nginx not found – skipping upstream swap (direct port expose mode)."
fi

# ── 5. Stop blue ──────────────────────────────────────────────────────────────
if docker ps -q -f name="$CONTAINER_BLUE" | grep -q .; then
  log "Stopping old blue container …"
  docker stop "$CONTAINER_BLUE"
  docker rm "$CONTAINER_BLUE"
fi

# Promote green → blue name for next deploy
docker rename "$CONTAINER_GREEN" "$CONTAINER_BLUE"

# Clean up dangling images
docker image prune -f >/dev/null

log "Deploy complete. Running: $(docker ps --filter name=$CONTAINER_BLUE --format '{{.Image}}')"
