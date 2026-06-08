# TSA Dashboard — VM deployment

Puts the dashboard behind your domain on an Ubuntu/Debian VM:

```
Browser  ──HTTPS──▶  Cloudflare edge  ──mTLS (origin pulls)──▶  nginx :443
                                                                  ├── /        →  dashboard/frontend/dist  (vite build)
                                                                  └── /api/*   →  uvicorn @ 127.0.0.1:8000
```

Cloudflare terminates TLS for the public; nginx terminates a second TLS connection from Cloudflare using a long-lived **Cloudflare Origin Certificate**, and rejects any client that doesn't present a cert signed by Cloudflare's **Authenticated Origin Pulls** root. Net effect: scrapers can't bypass Cloudflare by hitting the VM's IP directly.

The FastAPI backend runs as a systemd unit bound to `127.0.0.1:8000` — only nginx can reach it.

## 1. DNS

In Cloudflare, add **A records** on the apex and `www` pointing at the VM's public IP. Leave the proxy **on** (orange cloud) for both.

Then in **SSL/TLS → Overview**, set the encryption mode to **Full (strict)**.

## 2. Generate the Cloudflare Origin Certificate

Cloudflare dashboard → **SSL/TLS → Origin Server → Create Certificate**:

- Key type: **RSA (2048)**
- Hostnames: `your-domain.com` and `*.your-domain.com`
- Validity: 15 years (the cert is only ever shown to Cloudflare, so you don't need to rotate it like a public Let's Encrypt cert)

Copy the two blocks onto the VM:

```bash
sudo mkdir -p /etc/ssl/cloudflare
sudo chmod 700 /etc/ssl/cloudflare
sudo nano /etc/ssl/cloudflare/origin.pem   # paste "Origin Certificate"
sudo nano /etc/ssl/cloudflare/origin.key   # paste "Private key"
sudo chmod 600 /etc/ssl/cloudflare/origin.key
```

## 3. Enable Authenticated Origin Pulls

Cloudflare dashboard → **SSL/TLS → Origin Server → Authenticated Origin Pulls** → enable for your zone. `setup.sh` will download Cloudflare's published root CA into `/etc/ssl/cloudflare/authenticated_origin_pull_ca.pem` automatically.

## 4. Run the bootstrap script

On the VM, from the repo root:

```bash
sudo bash deploy/setup.sh
```

It will:

1. `apt install nginx`
2. `npm install + vite build` the frontend (if `dist/` is missing)
3. Pull the current Cloudflare edge IP list into `/etc/nginx/conf.d/cloudflare-ips.conf` so `set_real_ip_from` is accurate
4. Render `deploy/nginx/tsa-dashboard.conf.template` → `/etc/nginx/sites-available/tsa-dashboard.conf` and symlink into `sites-enabled/`
5. Render `deploy/systemd/tsa-dashboard-api.service.template` → `/etc/systemd/system/` and `systemctl enable --now` it
6. `nginx -t` and reload

Re-run the script any time you edit a template or want a fresh Cloudflare IP list.

## 5. Confirm

```bash
curl -I https://your-domain.com           # 200 OK, served by nginx
curl    https://your-domain.com/api/health  # JSON from FastAPI
systemctl status tsa-dashboard-api          # active (running)
journalctl -u tsa-dashboard-api -f          # backend logs
```

If `curl https://VM_PUBLIC_IP` from outside Cloudflare fails the TLS handshake, mTLS is working — that's the goal.

## Updating the frontend

When you ship a new build:

```bash
cd dashboard/frontend
npm run build      # writes dashboard/frontend/dist/
```

nginx picks it up immediately — no restart needed, and the fingerprinted `/assets/*` files are cached for a year while `index.html` is `no-cache`, so users get the new shell on next reload.

## Firewall (recommended)

If you're on AWS / GCP, lock the security group down to ports **80**, **443**, and **22**. Block 80/443 except from Cloudflare's [published IP ranges](https://www.cloudflare.com/ips/) for an additional layer beyond Authenticated Origin Pulls.

On the host itself:

```bash
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

## File map

```
deploy/
├── README.md                                  ← this file
├── setup.sh                                   ← run on the VM
├── nginx/
│   └── tsa-dashboard.conf.template            ← rendered to /etc/nginx/sites-available/
└── systemd/
    └── tsa-dashboard-api.service.template     ← rendered to /etc/systemd/system/
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 502 Bad Gateway on `/api/*` | uvicorn isn't up | `systemctl status tsa-dashboard-api`, then `journalctl -u tsa-dashboard-api -n 100` |
| 521 / 525 from Cloudflare | Origin cert missing or wrong | Re-paste `origin.pem` + `origin.key`, then `sudo nginx -t && sudo systemctl reload nginx` |
| 403 with "No required SSL certificate was sent" | Authenticated Origin Pulls toggle is on Cloudflare side but the CA file is missing | Re-run `setup.sh` (it downloads the CA) and re-enable in CF dashboard |
| Frontend loads but `/api/*` 404s | nginx site not enabled, or default site still bound | `ls /etc/nginx/sites-enabled/` — should have **only** `tsa-dashboard.conf` |
| Visitors all appear as Cloudflare IPs in logs | Either CF IP list is stale or `real_ip_header` isn't loaded — re-run `setup.sh` to refresh `/etc/nginx/conf.d/cloudflare-ips.conf` |
