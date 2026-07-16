#!/usr/bin/env bash
set -u

LOG_ROOT="${LOG_ROOT:-/var/log/vm_reboot_watch}"
STATE_ROOT="${STATE_ROOT:-/var/lib/vm_reboot_watch}"
SNAP_DIR="$LOG_ROOT/snapshots"
EVENT_DIR="$LOG_ROOT/events"
STATE_FILE="$STATE_ROOT/last_boot_id"
LOCK_FILE="$STATE_ROOT/watch.lock"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
RETENTION_MINUTES="${RETENTION_MINUTES:-65}"
HEAVY_EVERY="${HEAVY_EVERY:-12}"
MAX_PSTORE_FILES="${MAX_PSTORE_FILES:-5}"
HOST_ROLE="${HOST_ROLE:-auto}"

mkdir -p "$SNAP_DIR" "$EVENT_DIR" "$STATE_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

read_boot_id() {
  cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

is_physical_host() {
  case "$HOST_ROLE" in
    physical)
      return 0
      ;;
    vm)
      return 1
      ;;
  esac

  if [ -e /dev/ipmi0 ] || [ -d /sys/class/ipmi ]; then
    return 0
  fi
  return 1
}

capture_pressure() {
  local name
  for name in cpu memory io; do
    if [ -r "/proc/pressure/$name" ]; then
      echo "-- $name --"
      cat "/proc/pressure/$name" 2>&1 || true
    fi
  done
}

capture_meminfo_hotset() {
  awk '
    /^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active|Inactive|AnonPages|Mapped|Shmem|Slab|SReclaimable|SUnreclaim|KernelStack|PageTables|Dirty|Writeback|CommitLimit|Committed_AS|HugePages_Total|HugePages_Free):/ {
      print
    }
  ' /proc/meminfo 2>&1 || true
}

capture_pstore() {
  local dir path count
  dir="/sys/fs/pstore"
  if [ ! -d "$dir" ]; then
    return 0
  fi

  echo "== /sys/fs/pstore =="
  ls -l "$dir" 2>&1 || true
  count=0
  for path in "$dir"/*; do
    [ -e "$path" ] || continue
    count=$((count + 1))
    echo
    echo "-- $path --"
    head -n 200 "$path" 2>&1 || true
    if [ "$count" -ge "$MAX_PSTORE_FILES" ]; then
      break
    fi
  done
}

capture_daily_log_tails() {
  local stamp path
  for stamp in "$(date +%Y%m%d)" "$(date -d 'yesterday' +%Y%m%d)"; do
    path="/var/log/$stamp.log"
    if [ -f "$path" ]; then
      echo "== tail -n 200 $path =="
      tail -n 200 "$path" 2>&1 || true
      echo
    fi
  done
}

capture_ipmi_context() {
  if ! is_physical_host || ! has_cmd ipmitool; then
    return 0
  fi

  echo "== ipmitool chassis status =="
  ipmitool chassis status 2>&1 || true
  echo
  echo "== ipmitool mc info =="
  ipmitool mc info 2>&1 || true
  echo
  echo "== ipmitool sel elist last 120 =="
  ipmitool sel elist last 120 2>&1 || true
}

capture_libvirt_context() {
  if ! has_cmd virsh; then
    return 0
  fi

  echo "== virsh list --all =="
  virsh list --all 2>&1 || true
  echo
  echo "== journalctl -u libvirtd -n 120 =="
  journalctl -u libvirtd -n 120 --no-pager 2>&1 || true
}

current_boot_id="$(read_boot_id)"
previous_boot_id=""
if [ -f "$STATE_FILE" ]; then
  previous_boot_id="$(cat "$STATE_FILE" 2>/dev/null)"
fi

capture_boot_event() {
  local now boot_file
  now="$(date +%Y%m%dT%H%M%S%z)"
  boot_file="$EVENT_DIR/boot-change-$now.log"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "current_boot_id=$current_boot_id"
    echo "previous_boot_id=${previous_boot_id:-none}"
    echo
    echo "== who -b =="
    who -b 2>&1 || true
    echo
    echo "== uptime =="
    uptime 2>&1 || true
    echo
    echo "== last -x -n 20 =="
    last -x -n 20 2>&1 || true
    echo
    echo "== journalctl -b -1 -n 200 =="
    journalctl -b -1 -n 200 --no-pager 2>&1 || true
    echo
    echo "== journalctl -k -b -1 -n 200 =="
    journalctl -k -b -1 -n 200 --no-pager 2>&1 || true
    echo
    echo "== journalctl -b 0 -n 120 =="
    journalctl -b 0 -n 120 --no-pager 2>&1 || true
    echo
    echo "== journalctl -k -b 0 -n 120 =="
    journalctl -k -b 0 -n 120 --no-pager 2>&1 || true
    echo
    echo "== journalctl --list-boots =="
    journalctl --list-boots 2>&1 || true
    echo
    echo "== dmesg -T tail =="
    dmesg -T 2>&1 | tail -n 200 || true
    echo
    capture_daily_log_tails
    capture_pstore
    echo
    capture_libvirt_context
    echo
    capture_ipmi_context
  } >"$boot_file"
}

if [ -n "$previous_boot_id" ] && [ "$previous_boot_id" != "$current_boot_id" ]; then
  capture_boot_event
fi
printf '%s\n' "$current_boot_id" >"$STATE_FILE"

loop_count=0

while true; do
  current_boot_id="$(read_boot_id)"
  timestamp="$(date +%Y%m%dT%H%M%S%z)"
  sample_file="$SNAP_DIR/$timestamp.log"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "boot_id=$current_boot_id"
    echo
    echo "== uptime =="
    uptime 2>&1 || true
    echo
    echo "== who -b =="
    who -b 2>&1 || true
    echo
    echo "== /proc/loadavg =="
    cat /proc/loadavg 2>&1 || true
    echo
    echo "== /proc/uptime =="
    cat /proc/uptime 2>&1 || true
    echo
    echo "== pressure =="
    capture_pressure
    echo
    echo "== free -m =="
    free -m 2>&1 || true
    echo
    echo "== meminfo hotset =="
    capture_meminfo_hotset
    echo
    echo "== vmstat -SM =="
    vmstat -SM 2>&1 || true
    echo
    echo "== df -h =="
    df -h 2>&1 || true
    echo
    echo "== df -ih =="
    df -ih 2>&1 || true
    echo
    echo "== D state count =="
    ps -eo state= 2>&1 | awk '/^D/{c++} END{print c+0}' || true
    echo
    echo "== Z state count =="
    ps -eo state= 2>&1 | awk '/^Z/{c++} END{print c+0}' || true
    echo
    echo "== top cpu =="
    ps -eo pid,ppid,comm,%cpu,%mem,state --sort=-%cpu 2>&1 | head -n 25 || true
    echo
    echo "== top mem =="
    ps -eo pid,ppid,comm,%cpu,%mem,state --sort=-%mem 2>&1 | head -n 25 || true
  } >"$sample_file"

  loop_count=$((loop_count + 1))
  if [ $((loop_count % HEAVY_EVERY)) -eq 0 ]; then
    heavy_file="$EVENT_DIR/heavy-$timestamp.log"
    {
      echo "timestamp=$(date --iso-8601=seconds)"
      echo "boot_id=$current_boot_id"
      echo
      echo "== top -b -n 1 =="
      top -b -n 1 2>&1 | head -n 80 || true
      echo
      echo "== dmesg -T tail =="
      dmesg -T 2>&1 | tail -n 120 || true
      echo
      echo "== journalctl -p warning -n 120 =="
      journalctl -p warning -n 120 --no-pager 2>&1 || true
      echo
      echo "== journalctl -k -p warning -n 120 =="
      journalctl -k -p warning -n 120 --no-pager 2>&1 || true
      echo
      capture_libvirt_context
      echo
      capture_ipmi_context
    } >"$heavy_file"

    find "$LOG_ROOT" -type f -mmin +"$RETENTION_MINUTES" -delete 2>/dev/null || true
  fi

  sleep "$SAMPLE_INTERVAL"
done
