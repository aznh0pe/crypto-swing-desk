#!/usr/bin/env bash
# Keeps the `data` branch fresh around the clock: a pass every 10 minutes for about 5.5 hours, then this run
# starts its own successor and ends (GitHub stops any job at 6 hours). GitHub's scheduled triggers proved far too
# sparse to rely on (3 runs in 7 hours on the first day), so the schedule is only the watchdog that restarts
# this loop if it ever dies.
# Env: MODE (auto | history | once), PUSH_URL (authenticated repo URL), GH_TOKEN (for the hand-over), REPO.
set -u
MODE="${MODE:-auto}"
START=$(date +%s)
LIMIT=$(( ${LOOP_LIMIT_MIN:-330} * 60 ))
FAILS=0

pass() {
  for attempt in 1 2 3; do
    OUT_DIR=out python pipeline/fetch_data.py "$1" && return 0
    echo "fetch attempt $attempt failed, retrying in 30s"; sleep 30
  done
  return 1
}

publish() {
  # One commit with no parent, built in the checkout's own repository on top of what it already knows the branch
  # holds, so a push sends only the files that changed (the small recent/hourly files, not the full history).
  (
    set -e
    root="$PWD"
    export GIT_INDEX_FILE="$root/.git/data-index"
    rm -f "$GIT_INDEX_FILE"
    git --git-dir="$root/.git" --work-tree="$root/out" -C "$root/out" add -A .
    tree=$(git --git-dir="$root/.git" write-tree)
    commit=$(git --git-dir="$root/.git" -c user.name="crypto-swing-desk-bot" \
      -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
      commit-tree "$tree" -m "data $(date -u +%Y-%m-%dT%H:%MZ)")
    for attempt in 1 2 3; do
      if git --git-dir="$root/.git" push -q -f "$PUSH_URL" "$commit:refs/heads/data"; then
        git --git-dir="$root/.git" update-ref "refs/remotes/origin/data" "$commit"
        exit 0
      fi
      sleep 10
    done
    exit 1
  )
}

one() {
  if pass "$1" && publish; then FAILS=0; echo "published $(date -u +%H:%MZ)"; else FAILS=$((FAILS + 1)); echo "::warning::pass failed ($FAILS in a row)"; fi
}

one "$MODE"
[ "$MODE" = "auto" ] || exit $(( FAILS > 0 ))
while true; do
  if [ $(( $(date +%s) - START )) -ge "$LIMIT" ]; then
    # A workflow_dispatch made with the run's own token does start a new run. It waits in the concurrency group
    # until this one ends, so there is never a gap and never two loops.
    for attempt in 1 2 3; do
      gh workflow run crypto-data.yml --repo "$REPO" --ref main -f mode=auto && { echo "handed over to a new run"; exit 0; }
      sleep 20
    done
    echo "::warning::hand-over failed; the watchdog restarts the loop within minutes"
    exit 0
  fi
  # new feed code on main: end this run so its successor (already queued by the push) picks it up
  if git fetch -q --depth 1 origin main 2>/dev/null && ! git diff --quiet HEAD FETCH_HEAD -- pipeline .github/workflows/crypto-data.yml 2>/dev/null; then
    gh workflow run crypto-data.yml --repo "$REPO" --ref main -f mode=auto || true
    echo "feed code changed on main: handing over"; exit 0
  fi
  if [ "$FAILS" -ge 6 ]; then echo "::error::six passes failed in a row"; exit 1; fi
  now=$(date +%s); sleep $(( 600 - now % 600 + 45 ))   # 45 seconds past every 10-minute mark (00:00:45 UTC catches the daily close)
  one auto
done
