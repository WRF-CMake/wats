#!/bin/bash
set -e

# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.
# Note: This script is used from the WRF/WPS repositories during CI.

tmpl='{
  "request": {
    "branch": "%s",
    "config": {
      "env": {
        "global": [
          "TRIGGER_REPO=%s",
          "TRIGGER_COMMIT=%s",
          "TRIGGER_BUILD_NUMBER=%s",
          "WRF_REPO=%s",
          "WPS_REPO=%s",
          "WRF_COMMIT=%s",
          "WPS_COMMIT=%s",
          "DATA_REPO=%s",
          "WATS_MODE=%s"
        ]
      }
    }
  }
}'

body=$(printf "$tmpl" "$WATS_BRANCH" \
                      "$TRAVIS_REPO_SLUG" "$TRAVIS_COMMIT" "$TRAVIS_BUILD_NUMBER" \
                      "$WRF_REPO" "$WPS_REPO" "$WRF_COMMIT" "$WPS_COMMIT" \
                      "$WATS_DATA_REPO" "$WATS_MODE")

echo "Calling Travis CI API"
echo "Payload:"
echo $body

WATS_REPO_ENC="${WATS_REPO//\//%2F}"

code=$(curl -s -S -o /tmp/http.log -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Travis-API-Version: 3" \
    -H "Authorization: token ${TRAVIS_API_TOKEN}" \
    -d "$body" \
    https://api.travis-ci.org/repo/$WATS_REPO_ENC/requests)
    # TODO change to .com after migration

printf "\nStatus Code: $code\nResponse:\n"

cat /tmp/http.log

if [[ $code != 20* ]]; then
  exit 1
fi
