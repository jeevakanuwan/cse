"""
deploy.py — AWS EC2 deployment for CSE Stock Prediction App

Usage:
  py deploy.py setup    # First-time: create EC2 instance and configure server
  py deploy.py deploy   # Push latest code to server + restart app
  py deploy.py token    # Push updated CSE token from config.py to server
  py deploy.py status   # Show if app is running on server
  py deploy.py ssh      # Print SSH command to connect to server

Prerequisites (run once on your PC):
  py -m pip install boto3 paramiko
  aws configure          (enter your AWS Access Key, Secret, region ap-southeast-1)
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — change these if needed
# ---------------------------------------------------------------------------
AWS_REGION     = "ap-southeast-1"   # Singapore (closest to Sri Lanka)
INSTANCE_TYPE  = "t2.micro"         # Free tier eligible
KEY_NAME       = "cse-app-key"
SG_NAME        = "cse-app-sg"
APP_DIR        = "/opt/cse-app"
REMOTE_USER    = "ubuntu"
KEY_FILE       = Path("deploy") / "cse-app-key.pem"
STATE_FILE     = Path("deploy") / ".ec2-instance-id"

# Files/folders NOT copied to server (sensitive or large)
EXCLUDE = ["data/", "models/", "__pycache__", ".git", ".github", "*.pyc",
           "notebooks/", "tests/", "deploy/", ".gitignore"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run:  py -m pip install boto3 paramiko")
        sys.exit(1)


def _ec2():
    b = _boto3()
    return b.client("ec2", region_name=AWS_REGION)


def _get_instance_id() -> str | None:
    if STATE_FILE.exists():
        return STATE_FILE.read_text().strip()
    return None


def _save_instance_id(instance_id: str):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(instance_id)


def _get_instance_ip(instance_id: str) -> str | None:
    ec2 = _ec2()
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    instance = resp["Reservations"][0]["Instances"][0]
    return instance.get("PublicIpAddress")


def _wait_for_ssh(ip: str, timeout: int = 180):
    import socket
    print(f"Waiting for SSH on {ip} ", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((ip, 22), timeout=5):
                print(" ready.")
                return
        except OSError:
            print(".", end="", flush=True)
            time.sleep(5)
    print("\nERROR: SSH did not become available in time.")
    sys.exit(1)


def _ssh(ip: str, command: str, check: bool = True):
    """Run a command on the remote server via SSH."""
    cmd = [
        "ssh", "-i", str(KEY_FILE),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{REMOTE_USER}@{ip}",
        command,
    ]
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def _scp_to(ip: str, local: str, remote: str):
    """Copy a local file to the server."""
    cmd = [
        "scp", "-i", str(KEY_FILE),
        "-o", "StrictHostKeyChecking=no",
        local, f"{REMOTE_USER}@{ip}:{remote}",
    ]
    subprocess.run(cmd, check=True)


def _rsync_to(ip: str, local_dir: str, remote_dir: str):
    """Sync local directory to server (skips excluded paths)."""
    exclude_args = []
    for ex in EXCLUDE:
        exclude_args += ["--exclude", ex]

    cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -i {KEY_FILE} -o StrictHostKeyChecking=no",
        *exclude_args,
        f"{local_dir}/",
        f"{REMOTE_USER}@{ip}:{remote_dir}/",
    ]
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
    except FileNotFoundError:
        pass  # rsync not installed on Windows — use SCP fallback

    _scp_fallback(ip, local_dir, remote_dir)


def _scp_fallback(ip: str, local_dir: str, remote_dir: str):
    """Windows fallback: SCP individual files when rsync is unavailable."""
    local = Path(local_dir)
    skip_prefixes = tuple(EXCLUDE)
    files = []
    for f in local.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local)
            parts = rel.parts
            if any(str(rel).startswith(ex.rstrip("/")) for ex in EXCLUDE):
                continue
            if any(p.startswith("__") for p in parts):
                continue
            files.append(f)

    # create remote directory structure
    dirs = {str(f.relative_to(local).parent) for f in files if f.parent != local}
    for d in sorted(dirs):
        if d != ".":
            _ssh(ip, f"mkdir -p {remote_dir}/{d.replace(chr(92), '/')}")

    for f in files:
        rel = str(f.relative_to(local)).replace("\\", "/")
        _scp_to(ip, str(f), f"{remote_dir}/{rel}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup():
    """First-time: create EC2 instance, configure server, deploy code."""
    ec2 = _ec2()
    print("=== CSE App — AWS Setup ===\n")

    # 1. Key pair
    if KEY_FILE.exists():
        print(f"Key file {KEY_FILE} already exists — skipping key creation.")
    else:
        print(f"Creating key pair '{KEY_NAME}' ...")
        try:
            resp = ec2.create_key_pair(KeyName=KEY_NAME)
            KEY_FILE.parent.mkdir(exist_ok=True)
            KEY_FILE.write_text(resp["KeyMaterial"])
            # restrict permissions (required by SSH on Linux/Mac)
            if sys.platform != "win32":
                os.chmod(KEY_FILE, 0o400)
            print(f"Key saved to {KEY_FILE}")
        except ec2.exceptions.ClientError as e:
            if "InvalidKeyPair.Duplicate" in str(e):
                print(f"Key pair '{KEY_NAME}' already exists in AWS. "
                      "If you lost the .pem file, delete the key pair in the AWS console and rerun.")
                sys.exit(1)
            raise

    # 2. Security group
    sg_id = None
    try:
        resp = ec2.create_security_group(
            GroupName=SG_NAME,
            Description="CSE Stock Prediction App",
        )
        sg_id = resp["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": 22,  "ToPort": 22,  "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 80,  "ToPort": 80,  "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ],
        )
        print(f"Security group created: {sg_id}")
    except Exception as e:
        if "already exists" in str(e) or "Duplicate" in str(e):
            resp = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"Security group already exists: {sg_id}")
        else:
            raise

    # 3. Find Ubuntu 22.04 LTS AMI
    print("Finding Ubuntu 22.04 LTS AMI ...")
    resp = ec2.describe_images(
        Owners=["099720109477"],   # Canonical (Ubuntu official)
        Filters=[
            {"Name": "name",                  "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state",                 "Values": ["available"]},
            {"Name": "architecture",          "Values": ["x86_64"]},
            {"Name": "virtualization-type",   "Values": ["hvm"]},
        ],
    )
    ami = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)[0]
    ami_id = ami["ImageId"]
    print(f"Using AMI: {ami_id} ({ami['Name']})")

    # 4. Launch instance
    print(f"Launching {INSTANCE_TYPE} instance ...")
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        MinCount=1, MaxCount=1,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "cse-stock-app"}],
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    _save_instance_id(instance_id)
    print(f"Instance launched: {instance_id}")

    # 5. Wait for running state
    print("Waiting for instance to start ", end="", flush=True)
    while True:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state == "running":
            ip = resp["Reservations"][0]["Instances"][0]["PublicIpAddress"]
            print(f" running! IP: {ip}")
            break
        print(".", end="", flush=True)
        time.sleep(5)

    # 6. Wait for SSH
    _wait_for_ssh(ip)
    time.sleep(5)   # extra buffer for SSH daemon to fully start

    # 7. Copy systemd service files
    print("Copying service files ...")
    _scp_to(ip, "deploy/cse-app.service",       "/tmp/cse-app.service")
    _scp_to(ip, "deploy/cse-scheduler.service", "/tmp/cse-scheduler.service")

    # 8. Run server setup
    print("Running server setup (installs Python, nginx, etc.) ...")
    _scp_to(ip, "deploy/server_setup.sh", "/tmp/server_setup.sh")
    _ssh(ip, "chmod +x /tmp/server_setup.sh && sudo /tmp/server_setup.sh")

    # 9. Deploy app code
    print("\nDeploying application code ...")
    _rsync_to(ip, ".", APP_DIR)
    _ssh(ip, f"sudo chown -R {REMOTE_USER}:{REMOTE_USER} {APP_DIR}")

    # 10. Start services
    print("Starting services ...")
    _ssh(ip, "sudo systemctl start cse-app cse-scheduler")

    print(f"""
=== Setup complete! ===

Dashboard: http://{ip}
SSH:       ssh -i {KEY_FILE} {REMOTE_USER}@{ip}

Next steps:
  1. Run bootstrap:  py deploy.py ssh  then:
     python -c "from src.scraper import bootstrap_all_securities; bootstrap_all_securities(years=3)"
  2. When token expires, update config.py then:  py deploy.py token
""")


def cmd_deploy():
    """Push latest code to server and restart the app."""
    instance_id = _get_instance_id()
    if not instance_id:
        print("No instance found. Run:  py deploy.py setup")
        sys.exit(1)

    ip = _get_instance_ip(instance_id)
    print(f"Deploying to {ip} ...")
    _rsync_to(ip, ".", APP_DIR)
    _ssh(ip, f"sudo chown -R {REMOTE_USER}:{REMOTE_USER} {APP_DIR}")
    _ssh(ip, "sudo systemctl restart cse-app cse-scheduler")
    print("Deploy complete. App restarted.")


def cmd_token():
    """Push the CSE_TOKEN from local config.py to the server."""
    instance_id = _get_instance_id()
    if not instance_id:
        print("No instance found. Run:  py deploy.py setup")
        sys.exit(1)

    try:
        import config
        token = getattr(config, "CSE_TOKEN", "").split(";")[0].strip()
    except ImportError:
        print("config.py not found.")
        sys.exit(1)

    if not token or token == "eyJ...":
        print("No valid token in config.py. Paste a fresh token first.")
        sys.exit(1)

    ip = _get_instance_ip(instance_id)
    print(f"Pushing token to {ip} ...")
    # Write the token directly into config.py on the server
    escaped = token.replace('"', '\\"')
    _ssh(ip, (
        f'sudo sed -i \'s|CSE_TOKEN = ".*"|CSE_TOKEN = "{escaped}"|\' '
        f'{APP_DIR}/config.py'
    ))
    _ssh(ip, "sudo systemctl restart cse-app cse-scheduler")
    print("Token updated and services restarted.")


def cmd_status():
    """Show service status on the server."""
    instance_id = _get_instance_id()
    if not instance_id:
        print("No instance found. Run:  py deploy.py setup")
        sys.exit(1)

    ip = _get_instance_ip(instance_id)
    print(f"Server: {ip}\n")
    _ssh(ip, "sudo systemctl status cse-app --no-pager -l", check=False)
    print()
    _ssh(ip, "sudo systemctl status cse-scheduler --no-pager -l", check=False)


def cmd_ssh():
    """Print the SSH command to connect to the server."""
    instance_id = _get_instance_id()
    if not instance_id:
        print("No instance found. Run:  py deploy.py setup")
        sys.exit(1)

    ip = _get_instance_ip(instance_id)
    cmd = f"ssh -i {KEY_FILE} {REMOTE_USER}@{ip}"
    print(f"\nConnect with:\n  {cmd}\n")
    # On Windows try to open SSH directly
    if sys.platform == "win32":
        subprocess.run(cmd, shell=True)
    else:
        os.execvp("ssh", ["ssh", "-i", str(KEY_FILE), f"{REMOTE_USER}@{ip}"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "setup":  cmd_setup,
    "deploy": cmd_deploy,
    "token":  cmd_token,
    "status": cmd_status,
    "ssh":    cmd_ssh,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Commands:", ", ".join(COMMANDS))
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
