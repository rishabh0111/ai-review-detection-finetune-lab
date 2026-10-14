# Setting up the RTX 5090 machine (Windows + WSL2), driven over SSH

Goal: the 5090 laptop runs Ubuntu in WSL2 with GPU access, and the work laptop connects to it with
`ssh 5090` (and VS Code Remote-SSH). Everything below is done **on the 5090 machine** unless marked otherwise.
Check with office IT that running an SSH server on the machine is allowed.

## 1. NVIDIA driver (Windows side)

Install or update the normal **Windows** NVIDIA driver (Game Ready or Studio) for the RTX 5090. WSL2 uses the
Windows driver; **do not install an NVIDIA driver inside Ubuntu**.

## 2. Install WSL2 with Ubuntu

In **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when asked, then open "Ubuntu 24.04" from the Start menu and create a Linux user and password.

Check the GPU from inside Ubuntu:

```bash
nvidia-smi        # should list the RTX 5090
```

## 3. WSL settings: mirrored networking, systemd, no idle shutdown

Mirrored networking (Windows 11 22H2 or newer) makes services in WSL reachable on the Windows machine's own IP,
so no port forwarding is needed.

Create `C:\Users\<you>\.wslconfig` (Windows side):

```ini
[wsl2]
networkingMode=mirrored
memory=48GB
vmIdleTimeout=-1
```

`memory` caps WSL's share of the 64 GB RAM. `vmIdleTimeout=-1` stops the WSL VM shutting down when idle, but
Ubuntu itself can still stop shortly after the last terminal closes, taking sshd with it. Keep one WSL process
running with the log-on task in section 7.

Inside Ubuntu, enable systemd so the SSH server starts with WSL:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
```

Then in PowerShell: `wsl --shutdown`, and open Ubuntu again.

## 4. SSH server inside Ubuntu

Use port **2222** so it doesn't clash with a Windows OpenSSH server on 22.

```bash
sudo apt update && sudo apt install -y openssh-server
sudo sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
sudo systemctl enable --now ssh
sudo systemctl status ssh --no-pager    # should be active (running)
```

## 5. Allow the port through the Windows firewall

In **PowerShell as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "WSL SSH 2222" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow
# Mirrored mode also filters through the Hyper-V firewall; allow inbound for WSL:
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

Find the machine's IP: `ipconfig` (the Wi-Fi or Ethernet IPv4 address).

## 6. Key login from the work laptop

On the **work laptop** (Git Bash or PowerShell):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_5090 -C "work-laptop -> 5090"
```

Copy the contents of `~/.ssh/id_ed25519_5090.pub` into `~/.ssh/authorized_keys` on the 5090's Ubuntu:

```bash
# on the 5090, inside Ubuntu
mkdir -p ~/.ssh && chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys      # paste the public key on one line
chmod 600 ~/.ssh/authorized_keys
```

On the **work laptop**, add to `~/.ssh/config`:

```
Host 5090
    HostName <5090-machine-IPv4>
    Port 2222
    User <your-ubuntu-user>
    IdentityFile ~/.ssh/id_ed25519_5090
    ServerAliveInterval 30
```

`nvidia-smi` lives in `/usr/lib/wsl/lib`, which non-interactive SSH commands don't have on PATH. Add it at the
**top** of `~/.bashrc` inside Ubuntu (Ubuntu's `.bashrc` returns early for non-interactive shells):

```bash
sed -i '1i export PATH="/usr/lib/wsl/lib:$PATH"' ~/.bashrc
```

Test: `ssh 5090 nvidia-smi`.

Once key login works, turn off password login inside Ubuntu:

```bash
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

## 7. Keep it running

- Keep the 5090 machine powered and not sleeping during runs (Windows power settings: never sleep when
  plugged in).
- Keep Ubuntu running so sshd stays up: add a Task Scheduler task "at log on" running
  `wsl.exe -d Ubuntu-24.04 --exec sleep infinity` (a command that exits, like `/bin/true`, starts Ubuntu and
  lets it stop again a few seconds later). Or keep an Ubuntu terminal open.
- Run long jobs inside `tmux` so they survive an SSH disconnect: `sudo apt install -y tmux`, then `tmux new -s train`.

## 8. If the work laptop can't reach the 5090

Check from the work laptop: `Test-NetConnection <5090-IPv4> -Port 2222`. If it fails while
`Test-NetConnection localhost -Port 2222` succeeds on the 5090, the problem is the network path.

**Work laptop on the 5090's Wi-Fi hotspot** (work laptop has a `192.168.137.x` address, 5090 has `192.168.137.1`).
The hotspot adapter isn't mirrored into WSL, so forward a port on it to sshd (PowerShell as Administrator, on the
5090; 2223 because 2222 is already taken by sshd in mirrored mode):

```powershell
netsh interface portproxy add v4tov4 listenaddress=192.168.137.1 listenport=2223 connectaddress=127.0.0.1 connectport=2222
New-NetFirewallRule -DisplayName "WSL SSH hotspot 2223" -Direction Inbound -Protocol TCP -LocalPort 2223 -Action Allow
```

Then use `HostName 192.168.137.1` and `Port 2223` in `~/.ssh/config`. This works only while the hotspot is on. With
both machines on the office LAN, use the 5090's Ethernet address and port 2222 instead.

**Office network blocks the connection:**

Install Tailscale inside Ubuntu (`curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`) and on
the work laptop, then use the machine's Tailscale name or `100.x` address as `HostName`. Check this is allowed by IT.

## 9. Project environment (inside Ubuntu, over SSH)

Work in the Linux filesystem (`~/`), not `/mnt/c/...` (much slower for many small files).

```bash
sudo apt install -y git build-essential python3.12-venv
git clone https://github.com/rishabh0111/ai-review-detection-finetune-lab.git ~/ai-review-detection-finetune-lab
cd ~/ai-review-detection-finetune-lab
python3.12 -m venv .venv && source .venv/bin/activate
# RTX 50 series (Blackwell) needs PyTorch built for CUDA 12.8 or newer:
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print((x@x).float().abs().mean())"
```

The project's own requirements get added to the repo when the code is written (see `TRACKER.md`).

## Checklist

- [ ] `nvidia-smi` inside Ubuntu shows the RTX 5090
- [ ] `ssh 5090 nvidia-smi` works from the work laptop with the key, no password
- [ ] The torch matmul test above prints the GPU name and a number
