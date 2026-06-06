#!/usr/bin/env bash

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null)
    if [ -n "$pids" ]; then
        kill -9 $pids && echo "Killed port $port (PID $pids)"
    else
        echo "Nothing on port $port"
    fi
}

kill_port 8000
kill_port 5173
