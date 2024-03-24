#!/bin/bash
set -ev

source ./config.sh

pushd ../flattened
$FFMPEG -r $ANIMATION_FRAME_RATE -i "%"${ANIMATION_BUILD_FRAMES_ZEROS}d.png \
	-tune film \
	-movflags +faststart \
	-vcodec libx264 \
	-profile:v high \
	-pix_fmt yuv420p \
	${ANIMATION_CODENAME}-$(date -I).mp4
popd
