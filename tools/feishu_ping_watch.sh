#!/usr/bin/env bash
set -u

WEBHOOK_URL="${WEBHOOK_URL:-https://open.feishu.cn/open-apis/bot/v2/hook/c4398863-d8c0-40a1-ac99-9dbc1f53aa60}"
PING_INTERVAL="${PING_INTERVAL:-5}"
PING_TIMEOUT="${PING_TIMEOUT:-1}"
ALERT_COOLDOWN="${ALERT_COOLDOWN:-600}"
MENTION_NAME="${MENTION_NAME:-魏奥康}"
MENTION_OPEN_ID="${MENTION_OPEN_ID:-}"
STATE_DIR="${STATE_DIR:-/tmp/feishu_ping_watch}"
OOPS_BIN="${OOPS_BIN:-oops}"

mkdir -p "$STATE_DIR"

notify_feishu() {
  local label="$1"
  local state_hint="$2"
  local level="$3"
  local text="$4"
  local now host payload body curl_status

  now="$(date '+%Y-%m-%d %H:%M:%S %z')"
  host="$(hostname)"

  if [ -n "$MENTION_OPEN_ID" ]; then
    payload=$(
      printf '%s' \
        "{\"msg_type\":\"post\",\"content\":{\"post\":{\"zh_cn\":{\"title\":\"PING监控告警\",\"content\":[[{\"tag\":\"at\",\"user_id\":\"$MENTION_OPEN_ID\",\"user_name\":\"$MENTION_NAME\"},{\"tag\":\"text\",\"text\":\" $text\\n目标: $label\\n机器: $host\\n时间: $now\\n状态: $state_hint\\n级别: $level\"}]]}}}}"
    )
  else
    payload=$(
      printf '%s' \
        "{\"msg_type\":\"text\",\"content\":{\"text\":\"@$MENTION_NAME $text\\n目标: $label\\n机器: $host\\n时间: $now\\n状态: $state_hint\\n级别: $level\\n提示: 未设置 MENTION_OPEN_ID，这条消息不会产生真实飞书@。\"}}"
    )
  fi

  body="$(curl -sS -X POST "$WEBHOOK_URL" -H 'Content-Type: application/json; charset=utf-8' -d "$payload")"
  curl_status=$?
  if [ "$curl_status" -ne 0 ]; then
    echo "[$(date '+%F %T')] feishu webhook request failed for $label: curl exit $curl_status" >&2
    return 1
  fi

  case "$body" in
    *'"StatusCode":0'*|*'"code":0'*)
      return 0
      ;;
    *)
      echo "[$(date '+%F %T')] feishu webhook returned unexpected body for $label: $body" >&2
      return 1
      ;;
  esac
}

monitor_one() {
  local target="$1"
  local label="$2"
  local state_file lock_file now_ts should_alert
  state_file="$STATE_DIR/${label//[^A-Za-z0-9._-]/_}.state"
  lock_file="$STATE_DIR/${label//[^A-Za-z0-9._-]/_}.lock"

  exec 9>"$lock_file"
  if ! flock -n 9; then
    echo "[$(date '+%F %T')] monitor already running for $label" >&2
    return 0
  fi

  LAST_STATUS="unknown"
  LAST_ALERT_TS="0"
  if [ -f "$state_file" ]; then
    # shellcheck disable=SC1090
    . "$state_file"
  fi

  save_state() {
    cat >"$state_file" <<EOF
LAST_STATUS="$LAST_STATUS"
LAST_ALERT_TS="$LAST_ALERT_TS"
EOF
  }

  while true; do
    now_ts="$(date +%s)"
    if ping -c 1 -W "$PING_TIMEOUT" "$target" >/dev/null 2>&1; then
      if [ "$LAST_STATUS" = "down" ]; then
        notify_feishu "$label" "恢复" "INFO" "ping 已恢复"
        LAST_ALERT_TS="$now_ts"
      fi
      LAST_STATUS="up"
      save_state
      sleep "$PING_INTERVAL"
      continue
    fi

    should_alert=0
    if [ "$LAST_STATUS" != "down" ]; then
      should_alert=1
    elif [ $((now_ts - LAST_ALERT_TS)) -ge "$ALERT_COOLDOWN" ]; then
      should_alert=1
    fi

    if [ "$should_alert" -eq 1 ]; then
      if notify_feishu "$label" "故障" "ERROR" "ping 不通"; then
        LAST_ALERT_TS="$now_ts"
      fi
    fi

    LAST_STATUS="down"
    save_state
    sleep "$PING_INTERVAL"
  done
}

trim() {
  printf '%s' "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

is_public_ipv4() {
  local value="$1"
  case "$value" in
    ''|*[!0-9.]*)
      return 1
      ;;
  esac

  if [ "$(printf '%s' "$value" | awk -F. 'NF==4{ok=1; for(i=1;i<=4;i++) if($i<0 || $i>255 || $i=="") ok=0; print ok; exit} {print 0}')" != "1" ]; then
    return 1
  fi

  case "$value" in
    10.*|127.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*|192.168.*)
      return 1
      ;;
  esac
  return 0
}

extract_field() {
  local text="$1"
  local field="$2"
  printf '%s\n' "$text" | sed -n "s/.*$field:[[:space:]]*//p" | head -n 1
}

resolve_targets_from_oops() {
  local source="$1"
  local info physical_internal virtual_internal physical_public virtual_public
  local ip raw_ip

  if is_public_ipv4 "$source"; then
    printf '%s|%s\n' "$source" "$source"
    return 0
  fi

  info="$("$OOPS_BIN" search "$source" 2>/dev/null)" || {
    echo "failed to resolve target via oops search: $source" >&2
    return 1
  }

  physical_internal="$(trim "$(extract_field "$info" "physical_internal_ip")")"
  virtual_internal="$(trim "$(extract_field "$info" "virtual_internal_ip")")"
  physical_public="$(trim "$(extract_field "$info" "physical_public_ip")")"
  virtual_public="$(trim "$(extract_field "$info" "virtual_public_ip")")"

  if [ "$source" = "$virtual_internal" ] && [ -n "$virtual_public" ]; then
    for raw_ip in $(printf '%s\n' "$virtual_public" | tr ',' '\n'); do
      ip="$(trim "$raw_ip")"
      [ -n "$ip" ] && printf '%s|%s => %s\n' "$ip" "$source" "$ip"
    done
    return 0
  fi

  if [ "$source" = "$physical_internal" ] && [ -n "$physical_public" ]; then
    printf '%s|%s => %s\n' "$physical_public" "$source" "$physical_public"
    return 0
  fi

  if [ -n "$virtual_public" ]; then
    for raw_ip in $(printf '%s\n' "$virtual_public" | tr ',' '\n'); do
      ip="$(trim "$raw_ip")"
      [ -n "$ip" ] && printf '%s|%s => %s\n' "$ip" "$source" "$ip"
    done
    return 0
  fi

  if [ -n "$physical_public" ]; then
    printf '%s|%s => %s\n' "$physical_public" "$source" "$physical_public"
    return 0
  fi

  echo "no public ip found via oops search for: $source" >&2
  return 1
}

parse_targets() {
  if [ "$#" -gt 0 ]; then
    printf '%s\n' "$@"
    return
  fi

  if [ -n "${TARGETS:-}" ]; then
    printf '%s\n' "$TARGETS" | tr ', ' '\n\n' | sed '/^$/d'
    return
  fi

  if [ -n "${TARGET:-}" ]; then
    printf '%s\n' "$TARGET"
    return
  fi

  return 1
}

main() {
  local sources=()
  local resolved=()
  local item target label source
  declare -A seen_labels=()

  if ! mapfile -t sources < <(parse_targets "$@"); then
    echo "usage: $0 <host-or-ip> [more-hosts...]" >&2
    echo "or set TARGET=host" >&2
    echo "or set TARGETS='ip1 ip2 ip3'" >&2
    exit 2
  fi

  for source in "${sources[@]}"; do
    while IFS= read -r item; do
      [ -n "$item" ] || continue
      resolved+=("$item")
    done < <(resolve_targets_from_oops "$source")
  done

  if [ "${#resolved[@]}" -eq 0 ]; then
    echo "no public targets resolved from oops search" >&2
    exit 1
  fi

  trap 'kill 0' INT TERM EXIT
  for item in "${resolved[@]}"; do
    target="${item%%|*}"
    label="${item#*|}"
    if [ -n "${seen_labels[$label]:-}" ]; then
      continue
    fi
    seen_labels[$label]=1
    monitor_one "$target" "$label" &
  done
  wait
}

main "$@"
