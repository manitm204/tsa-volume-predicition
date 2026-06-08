#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/logs"

# Kill any stale processes from a previous run so ports 8000 / 5173 are free
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "vite"             2>/dev/null
sleep 1

echo "Starting backend..."
cd "$SCRIPT_DIR/dashboard/backend"
uvicorn main:app --reload --port 8000 > "$SCRIPT_DIR/logs/backend.log" 2>&1 &

echo "Starting frontend..."
cd "$SCRIPT_DIR/dashboard/frontend"
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &

PUBLIC_IP="18.222.14.86"

echo "Backend:  http://localhost:8000           (logs/backend.log)"
echo "Frontend: http://localhost:5173           (logs/frontend.log)"
echo "Public:   http://$PUBLIC_IP:5173   ← open this in your browser"
