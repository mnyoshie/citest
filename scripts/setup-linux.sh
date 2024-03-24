
if [[ "$(uname -mso)" == "Linux "*" Android" ]]; then
	SUDO=
else
	SUDO=sudo
fi

PACKAGES=" curl"
PACKAGES+=" curl"
PACKAGES+=" imagemagick"
PACKAGES+=" ffmpeg"
PACKAGES+=" busybox"
PACKAGES+=" gzip"
PACKAGES+=" zstd"
PACKAGES+=" tar"
#PACKAGES+=" yt-dlp"
PACKAGES+=" gnupg"
yes | $SUDO apt install $PACKAGES
