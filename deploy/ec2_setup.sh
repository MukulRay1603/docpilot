#!/usr/bin/env bash
# Run once on a fresh EC2 instance (Amazon Linux 2023 or Ubuntu 22.04).
# Sets up Docker, AWS CLI v2, ECR credential helper, and the deploy script.
set -euo pipefail

APP_DIR="/opt/xr-qa"
DEPLOY_USER="ec2-user"   # change to "ubuntu" on Ubuntu AMIs

# ── Docker ────────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "Installing Docker …"
  if command -v dnf &>/dev/null; then
    dnf install -y docker
  else
    apt-get update && apt-get install -y docker.io
  fi
  systemctl enable --now docker
  usermod -aG docker "$DEPLOY_USER"
fi

# ── AWS CLI v2 ────────────────────────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
  echo "Installing AWS CLI v2 …"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp/aws-install
  /tmp/aws-install/aws/install
  rm -rf /tmp/awscli.zip /tmp/aws-install
fi

# ── ECR credential helper ──────────────────────────────────────────────────────
if ! command -v docker-credential-ecr-login &>/dev/null; then
  echo "Installing ECR credential helper …"
  VERSION=$(curl -s https://api.github.com/repos/awslabs/amazon-ecr-credential-helper/releases/latest \
    | grep tag_name | cut -d'"' -f4)
  curl -fsSL "https://amazon-ecr-credential-helper.s3.amazonaws.com/${VERSION}/linux-amd64/docker-credential-ecr-login" \
    -o /usr/local/bin/docker-credential-ecr-login
  chmod +x /usr/local/bin/docker-credential-ecr-login
fi

mkdir -p /root/.docker
cat > /root/.docker/config.json <<'EOF'
{ "credsStore": "ecr-login" }
EOF

# ── App directory ─────────────────────────────────────────────────────────────
mkdir -p "$APP_DIR/deploy"
cp "$(dirname "$0")/deploy.sh" "$APP_DIR/deploy/deploy.sh"
chmod +x "$APP_DIR/deploy/deploy.sh"

echo "EC2 setup complete. App directory: $APP_DIR"
