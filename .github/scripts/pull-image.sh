#!/usr/bin/env bash
# Pull an image, going through the in-cluster pull-through proxy for ghcr.io
# when the runner can reach it, and tag the result with the original name.
#
# docker.io pulls need nothing here: `docker pull` and `docker build` (the
# daemon's built-in BuildKit) already use dockerd's --registry-mirror. dockerd
# has no mirror setting for ghcr.io, so those pulls are rewritten to the proxy.
# If the proxy is unreachable (or dockerd does not allow it over plain HTTP),
# the pull falls back to ghcr.io and the job still works, just slower.
set -euo pipefail

image="$1"
proxy="${GHCR_PROXY:-ghcr-proxy.github-arc.svc.cluster.local:5000}"

echo "dockerd registry config: $(docker info --format '{{json .RegistryConfig.Mirrors}} insecure={{json .RegistryConfig.InsecureRegistryCIDRs}}' 2>/dev/null || echo unknown)"

case "$image" in
  ghcr.io/*) ;;
  *) exec docker pull "$image" ;;
esac

mirrored="${proxy}/${image#ghcr.io/}"
start=$SECONDS
if docker pull "$mirrored"; then
  docker tag "$mirrored" "$image"
  docker rmi "$mirrored" >/dev/null
  echo "Pulled $image via $proxy in $((SECONDS - start))s"
  exit 0
fi

echo "::warning::Could not pull via $proxy; pulling $image from ghcr.io directly"
start=$SECONDS
docker pull "$image"
echo "Pulled $image from ghcr.io in $((SECONDS - start))s"
