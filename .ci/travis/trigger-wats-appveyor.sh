#!/bin/bash
set -e

# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.
# Note: This script is used from the WRF/WPS repositories during CI.

WATS_REPO_NAME=`basename $WATS_REPO`

tmpl='{
  "accountName": "dmey",
  "projectSlug": "%s",
  "branch": "%s",
  "environmentVariables": {
    "TRIGGER_REPO": "%s",
    "TRIGGER_COMMIT": "%s",
    "TRIGGER_BUILD_NUMBER": "%s",
    "WRF_REPO": "%s",
    "WPS_REPO": "%s",
    "WRF_COMMIT": "%s",
    "WPS_COMMIT": "%s",
    "DATA_REPO": "%s",
    "WATS_MODE": "%s"
  }
}'

body=$(printf "$tmpl" "$WATS_REPO_USER" "$WATS_REPO_NAME" "$WATS_BRANCH" \
                      "$TRAVIS_REPO_SLUG" "$TRAVIS_COMMIT" "$TRAVIS_BUILD_NUMBER" \
                      "$WRF_REPO" "$WPS_REPO" "$WRF_COMMIT" "$WPS_COMMIT" \
                      "$WATS_DATA_REPO" "$WATS_MODE")

echo "Calling AppVeyor API"
echo "Payload:"
echo $body

code=$(curl -s -S -o http.log -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Authorization: Bearer ${APPVEYOR_API_TOKEN}" \
    -d "$body" \
    https://ci.appveyor.com/api/builds)

printf "\nStatus Code: $code\nResponse:\n"

cat http.log

if [[ $code != 20* ]]; then
  exit 1
fi
