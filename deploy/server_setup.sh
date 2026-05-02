#!/bin/bash
# One-time setup script — runs on fresh Ubuntu 22.04 EC2 instance
# Called automatically by deploy.py during first-time setup

set -e   # stop on any error

APP_DIR="/opt/cse-app"
VENV="$APP_DIR/venv"
USER="ubuntu"

echo "=== [1/6] Updating system packages ==="
apt-get update -y
apt-get install -y python3 python3-pip python3-venv nginx git

echo "=== [2/6] Creating app directory ==="
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/models"
chown -R "$USER:$USER" "$APP_DIR"

echo "=== [3/6] Creating Python virtual environment ==="
sudo -u "$USER" python3 -m venv "$VENV"
sudo -u "$USER" "$VENV/bin/pip" install --upgrade pip

echo "=== [4/6] Installing Python dependencies ==="
sudo -u "$USER" "$VENV/bin/pip" install \
    streamlit pandas numpy scikit-learn \
    requests beautifulsoup4 plotly \
    apscheduler joblib lxml

echo "=== [5/6] Configuring nginx ==="
cat > /etc/nginx/sites-available/cse-app << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

ln -sf /etc/nginx/sites-available/cse-app /etc/nginx/sites-enabled/cse-app
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx

echo "=== [6/6] Installing systemd services ==="
cp /tmp/cse-app.service       /etc/systemd/system/cse-app.service
cp /tmp/cse-scheduler.service /etc/systemd/system/cse-scheduler.service
systemctl daemon-reload
systemctl enable cse-app
systemctl enable cse-scheduler

echo ""
echo "=== Server setup complete ==="
echo "App will start automatically after deploy.py copies the code."
