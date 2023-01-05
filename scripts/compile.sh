#!/bin/bash
set -ev

source config.sh

pushd ../flattened
ffmpeg -r $frame_rate -i %04d.png \
	-crf 30 \
	-movflags +faststart \
	-vcodec libx264 \
	-pix_fmt yuv420p \
	twp-$(date -I).mp4
popd
