#!/usr/bin/env bash
# Usage: notarize.sh FILE [TARGET]: notarizes FILE, then staples its ticket onto TARGET (FILE by default).
# Briefcase notarizes only with a keychain profile; notarytool takes the API key.
set -euo pipefail

file=$1
target=${2:-$1}
key="$RUNNER_TEMP/notary-key.p8"
trap 'rm -f "$key"' EXIT
printf '%s\n' "$APP_STORE_CONNECT_KEY" > "$key"
credentials=(--key "$key" --key-id "$APP_STORE_CONNECT_KEY_ID" --issuer "$APP_STORE_CONNECT_ISSUER_ID")
result="$RUNNER_TEMP/notarization.txt"
# notarytool's JSON can come on either stream: after a timeout, on stderr.
xcrun notarytool submit "$file" "${credentials[@]}" --wait --timeout 2h --output-format json > "$result" 2>&1 || true
cat "$result"
field() { jq -rR "fromjson? | objects | .$1 // empty" "$result" | tail -n 1; }
id=$(field id)
status=$(field status)
echo "id=$id" >> "$GITHUB_OUTPUT"
echo "status=$status" >> "$GITHUB_OUTPUT"
case "$status" in
  Accepted) ;;
  Invalid | Rejected)
    echo "::error::Apple marked submission $id $status."
    xcrun notarytool log "$id" "${credentials[@]}"
    exit 1
    ;;
  *)
    if [ -n "$id" ]; then
      echo "::error::Submission $id is still processing at Apple: $(field message)"
    else
      echo "::error::notarytool returned no submission."
    fi
    exit 1
    ;;
esac
# A ticket can take a moment to be served after its acceptance.
for attempt in 1 2 3; do
  xcrun stapler staple "$target" && break
  [ "$attempt" -lt 3 ] || exit 1
  sleep 30
done
