#!/usr/bin/env bash
set -euo pipefail
# the runs below need links and the communicator on, whatever the caller set
unset GGML_RPC_NO_LINK GGML_RPC_NO_COMM

server=$1
client=$2
# below the ports Linux (from 32768) and macOS (from 49152) pick for outgoing connections, any of which can hold a
# port the servers need: their own, and those + 1000 for links and COMM_INIT
port_a=$((20000 + $$ % 10000))
port_b=$((port_a + 1))
endpoint_a="127.0.0.1:${port_a}"
endpoint_b="127.0.0.1:${port_b}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid_a:-}" "${pid_b:-}" "${pid_client:-}" 2>/dev/null || true
    # a stopped server acts on the signal only once it runs again
    kill -CONT "${pid_a:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

# waits up to 30 s for port $1, while process $2 runs
wait_for_port() {
    local port=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        if ! kill -0 "$2" 2>/dev/null; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

# waits up to 30 s for file $1, while process $2 runs
wait_for_file() {
    for _ in {1..3000}; do
        if [ -e "$1" ]; then
            return 0
        fi
        if ! kill -0 "$2" 2>/dev/null; then
            [ -e "$1" ]
            return $?
        fi
        sleep 0.01
    done
    return 1
}

# prints $1, the servers' logs and the files given after $1, and fails
fail() {
    echo "$1"
    shift
    cat "$test_dir/server-a.log" "$test_dir/server-b.log" "$@"
    exit 1
}

expect_log() {
    grep -qF "$2" "$test_dir/server-$1.log" || fail "server $1 did not log: $2"
}

# the ordering steps hang if the halves of a copy reach the servers out of order, so the client gets 300 s
run_client() {
    "$@" &
    pid_client=$!
    for _ in {1..3000}; do
        if ! kill -0 "$pid_client" 2>/dev/null; then
            wait "$pid_client" || fail "the client failed"
            unset pid_client
            return 0
        fi
        sleep 0.1
    done
    fail "the client did not finish in 300 s"
}

"$server" --device CPU --host 127.0.0.1 --port "$port_a" >"$test_dir/server-a.log" 2>&1 &
pid_a=$!
"$server" --device CPU --host 127.0.0.1 --port "$port_b" >"$test_dir/server-b.log" 2>&1 &
pid_b=$!
wait_for_port "$port_a" "$pid_a" || fail "server A did not listen on $endpoint_a"
wait_for_port "$port_b" "$pid_b" || fail "server B did not listen on $endpoint_b"

# before any link run, so the servers' logs hold no link lines yet
run_client env GGML_RPC_NO_LINK=1 "$client" "$endpoint_a" "$endpoint_b"
if grep -qE "link [0-9]+: (accepted on|connected to)" "$test_dir/server-a.log" "$test_dir/server-b.log"; then
    fail "a link was opened with GGML_RPC_NO_LINK set"
fi

run_client "$client" "$endpoint_a" "$endpoint_b"

if grep -q "invalid data ptr" "$test_dir/server-b.log"; then
    cat "$test_dir/server-b.log"
    exit 1
fi
# the links opened between the servers, link 0 from A to B and link 1 back
expect_log b "link 0: accepted on 127.0.0.1:$((port_b + 1000))"
expect_log a "link 0: connected to 127.0.0.1:$((port_b + 1000))"
expect_log a "link 1: accepted on 127.0.0.1:$((port_a + 1000))"
expect_log b "link 1: connected to 127.0.0.1:$((port_a + 1000))"
# the last step ends B's session while A's goes on, so its copy from A to B opens the link again, under the old id
if grep -qE "link ([2-9]|[1-9][0-9]+): " "$test_dir/server-a.log" "$test_dir/server-b.log"; then
    fail "a link opened again between the same two servers got a new id"
fi
[ "$(grep -cF "link 0: connected to" "$test_dir/server-a.log")" = 2 ] || fail "the link from A to B did not open again"
if grep -qF "TCP keepalive not fully set" "$test_dir/server-a.log" "$test_dir/server-b.log"; then
    fail "a link has no TCP keepalive"
fi

# server B exits while server A has a link to it: A's send on the link then fails, and A must keep running
client_log="$test_dir/client-peer-exit.log"
mkdir "$test_dir/peer-exit"
connected_a=$(grep -cF "link 0: connected to" "$test_dir/server-a.log" || true)
"$client" "$endpoint_a" "$endpoint_b" --peer-exit "$test_dir/peer-exit" >"$client_log" 2>&1 &
pid_client=$!
wait_for_file "$test_dir/peer-exit/linked" "$pid_client" || fail "the peer-exit client did not make its first copy" "$client_log"
[ "$(grep -cF "link 0: connected to" "$test_dir/server-a.log")" -gt "$connected_a" ] ||
    fail "the peer-exit client did not open its link" "$client_log"
# A stays stopped while the client posts the copy and B exits, so A sends it only to a closed link
closed_a=$(grep -cF "Client connection closed" "$test_dir/server-a.log" || true)
kill -STOP "$pid_a" || fail "server A exited before the copy" "$client_log"
touch "$test_dir/peer-exit/stopped"
wait_for_file "$test_dir/peer-exit/posted" "$pid_client" || fail "the peer-exit client did not post its copy" "$client_log"
kill -9 "$pid_b" || fail "server B exited before it was killed" "$client_log"
wait "$pid_b" 2>/dev/null || true
unset pid_b
kill -CONT "$pid_a" || fail "server A exited while stopped" "$client_log"
touch "$test_dir/peer-exit/go"
client_ok=1
wait "$pid_client" || client_ok=0
unset pid_client
[ "$client_ok" = 1 ] || fail "the peer-exit client failed" "$client_log"
# A ends the session once the send fails, and must still run after that
for _ in {1..600}; do
    if [ "$(grep -cF "Client connection closed" "$test_dir/server-a.log" || true)" -gt "$closed_a" ] || ! kill -0 "$pid_a" 2>/dev/null; then
        break
    fi
    sleep 0.05
done
kill -0 "$pid_a" 2>/dev/null || fail "server A exited after its link peer did"
expect_log a "link 0: failed to send"
