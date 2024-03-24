#!/bin/bash
set -eo pipefail

source ./config.sh
source ../cuts.sh


# default src format
BUILD_FRAMES_SRCFORMAT="tar.zst"

for c in ${cuts[@]}; do
	(
	pushd ../$c > /dev/null
	source build.sh
	animation_pre_src_download

	# This is what we're only going to support throughout this
	# upcoming future years. NO MORE. NO LESS.

	# the use of zip is highly discouraged since it's used
	# result to SIGPIPE. Better to use tar.
	if [[ "$BUILD_FRAMES_SRCFORMAT" == "zip" ]]; then
		echo "CURL $c | VSHA256SUM > >(BUSYBOX UNZIP)"
		$CURL -sL "$BUILD_FRAMES_SRCURL" | \
			$VSHA256SUM $BUILD_FRAMES_SRCSHA256 > >(dd | $BUSYBOX unzip -)
	elif [[ "$BUILD_FRAMES_SRCFORMAT" == "tar.zst" ]]; then
		echo "CURL $c | VSHA256SUM > >(TAR ZSTD)"
		$CURL -sL "$BUILD_FRAMES_SRCURL" | \
			$VSHA256SUM $BUILD_FRAMES_SRCSHA256 > >($TAR --zstd -xf -)
	elif [[ "$BUILD_FRAMES_SRCFORMAT" == "tar.gz" ]]; then
		echo "CURL $c | VSHA256SUM > >(TAR GZIP)"
		$CURL -sL "$BUILD_FRAMES_SRCURL" | \
			$VSHA256SUM $BUILD_FRAMES_SRCSHA256 > >($TAR --gzip -xvf -)
	else
		echo "invalid frames format: \"$BUILD_FRAMES_SRCFORMAT\""
		exit 1
	fi
	nonce=$( printf "$BUILD_FRAMES_DESCRIPTION ${BUILD_FRAMES_AUTHOR[@]}" | sha256sum )
	for author in "${BUILD_FRAMES_AUTHOR[@]}"; do
		echo "$BUILD_FRAMES_STARTING_FRAME-${nonce:0:16}  $author"
	done

	animation_post_src_download
	popd > /dev/null
	) &

	if [[ $(jobs -r -p | wc -l) -ge 1 ]]; then
		wait -n
	fi
done
wait

