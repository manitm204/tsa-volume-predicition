#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting backend..."
cd "$SCRIPT_DIR/dashboard/backend"
uvicorn main:app --reload --port 8000 > "$SCRIPT_DIR/logs/backend.log" 2>&1 &

echo "Starting frontend..."
cd "$SCRIPT_DIR/dashboard/frontend"
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &

echo "Backend:  http://localhost:8000  (logs/backend.log)"
echo "Frontend: http://localhost:5173  (logs/frontend.log)"
