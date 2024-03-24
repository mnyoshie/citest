ANIMATION_CODENAME="twp"

ANIMATION_FRAMES=38
ANIMATION_BUILD_FRAMES_ZEROS="04"
ANIMATION_FRAME_RATE=12

TAR=$(which tar)
CURL=$(which curl)
FFMPEG=$(which ffmpeg)
BUSYBOX=$(which busybox)
MAGICK=$(which magick || echo ) 
IM_CONVERT="$MAGICK convert"
IM_COMPOSITE="$MAGICK composite"

animation_pre_src_download() {
	:;
}

animation_post_src_download() {
	:;
}

if [[ "$(uname -mso)" == "Linux aarch64 Android" ]]; then
	VSHA256SUM=$(realpath bin/vsha256sum-aarch64-linux-android)
elif [[ "$(uname -mso)" == "Linux x86_64 GNU/Linux" ]]; then
	VSHA256SUM=$(realpath bin/vsha256sum-x86_64-linux-gnu)
else
	echo "unsupported machine for $(uname -mso)"
	exit 1
fi
