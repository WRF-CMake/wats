#!/bin/bash
set -e

# Copyright 2018 M. Riechert and D. Meyer. Licensed under the MIT License.

if [ "$PENDING" == "1" ]; then
    state="pending"
elif [ "$TRAVIS_TEST_RESULT" == "0" ] || [ "$SUCCESS" == "1" ]; then
    state="success"
else
    state="error"
fi

if [ "$APPVEYOR" == "True" ]; then
    build_url="https://ci.appveyor.com/project/$APPVEYOR_REPO_NAME/build/$APPVEYOR_BUILD_VERSION"
    context="windows/$MODE"
else
    build_url="https://travis-ci.com/$TRAVIS_REPO_SLUG/builds/$TRAVIS_BUILD_ID"
    context="$TRAVIS_OS_NAME/$MODE"
fi

tmpl='{
  "state": "%s",
  "target_url": "%s",
  "context": "wats/%s"
}'

body=$(printf "$tmpl" "$state" "$build_url" "$context")

echo "Calling GitHub Status API"
echo "Payload:"
echo $body

code=$(curl -s -S -o /tmp/http.log -w "%{http_code}" -u :$GITHUB_STATUS_TOKEN -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d "$body" \
    https://api.github.com/repos/$TRIGGER_REPO/statuses/$TRIGGER_COMMIT)

printf "\nStatus Code: $code\nResponse:\n"

cat /tmp/http.log

if [[ $code != 20* ]]; then
  exit 1
fi
