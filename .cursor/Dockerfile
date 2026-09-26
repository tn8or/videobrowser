# Cloud Agent base image for videobrowser.
# Mirrors Dockerfile.ci so agents match CI: Python 3.14 + Cairo build deps +
# ffmpeg + git. Keep the FROM in sync with /.python-version and Dockerfile.ci.
FROM python:3.14-bookworm@sha256:bfb689a7986adc6d5f16722e06c78e755efe4062e56276715fef450cbad09436

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libcairo2-dev \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
