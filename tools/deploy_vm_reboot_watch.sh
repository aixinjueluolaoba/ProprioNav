#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SRC="$ROOT_DIR/vm_reboot_watch.sh"
SERVICE_SRC="$ROOT_DIR/vm_reboot_watch.service"

VM_TARGETS=(
  "61.155.154.95"
  "61.155.154.97"
  "61.155.154.99"
  "61.155.154.101"
)
PHYSICAL_TARGET="10.132.97.44"
PHYSICAL_PUBLIC_TARGET="121.46.195.174"

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=("${VM_TARGETS[@]}" "$PHYSICAL_PUBLIC_TARGET")
fi

run_remote() {
  local target="$1"
  shift
  rtk oops "$target" "$*"
}

push_file() {
  local target="$1"
  local local_path="$2"
  local remote_path="$3"
  rtk oops push "$target" "$local_path" "$remote_path"
}

host_role_for() {
  local target="$1"
  if [ "$target" = "$PHYSICAL_TARGET" ] || [ "$target" = "$PHYSICAL_PUBLIC_TARGET" ]; then
    printf '%s\n' "physical"
  else
    printf '%s\n' "vm"
  fi
}

deploy_one() {
  local target="$1"
  local role
  local role_file
  role="$(host_role_for "$target")"
  role_file="$(mktemp)"
  printf '[Service]\nEnvironment=HOST_ROLE=%s\n' "$role" >"$role_file"

  printf '[%s] pushing files\n' "$target"
  push_file "$target" "$WATCH_SRC" /tmp/vm_reboot_watch.sh
  push_file "$target" "$SERVICE_SRC" /tmp/vm_reboot_watch.service
  push_file "$target" "$role_file" /tmp/vm_reboot_watch_role.conf

  printf '[%s] installing watcher\n' "$target"
  run_remote "$target" "install -m 0755 /tmp/vm_reboot_watch.sh /usr/local/bin/vm_reboot_watch.sh"
  run_remote "$target" "install -m 0644 /tmp/vm_reboot_watch.service /etc/systemd/system/vm_reboot_watch.service"
  run_remote "$target" "mkdir -p /etc/systemd/system/vm_reboot_watch.service.d"
  run_remote "$target" "install -m 0644 /tmp/vm_reboot_watch_role.conf /etc/systemd/system/vm_reboot_watch.service.d/role.conf"
  run_remote "$target" "systemctl daemon-reload"
  run_remote "$target" "systemctl enable vm_reboot_watch.service"
  run_remote "$target" "systemctl restart vm_reboot_watch.service"
  run_remote "$target" "systemctl is-active vm_reboot_watch.service"
  run_remote "$target" "head -n 5 /usr/local/bin/vm_reboot_watch.sh"
  rm -f "$role_file"
}

for target in "${TARGETS[@]}"; do
  deploy_one "$target"
done
