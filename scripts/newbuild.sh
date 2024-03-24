#!/bin/bash

# XXX You will only have to modify BUILD_* to your liking.
# this will create a new directory 
BUILD_FRAMES_AUTHOR=(
"YOUR NAME"
"ANOTHER NAME 1"
"ANOTHER AUTHOR2"
)
BUILD_FRAMES_DESCRIPTION="Short description about the animation or its scene"

# the first frame that appears in your animation with
# the zeroes padding included
BUILD_FRAMES_STARTING_FRAME="0000"

# url to retrieve the frames
BUILD_FRAMES_SRCURL="https://example.com/file.tar.zst"

# What file format is the frames stored into? (zip, tar.zst, tar.gz)
BUILD_FRAMES_SRCFORMAT="tar.zst"

# sha256sum of the source (tar/zip)
BUILD_FRAMES_SRCSHA256=""

# license
BUILD_FRAMES_LICENSE="CC0"

nonce=$( printf "$BUILD_FRAMES_DESCRIPTION ${BUILD_FRAMES_AUTHOR[@]}" | sha256sum )
builddir="$BUILD_FRAMES_STARTING_FRAME-${nonce:0:16}"
mkdir ../$builddir

echo "BUILD_FRAMES_AUTHOR=("                                         >> ../$builddir/build.sh
for author in "${BUILD_FRAMES_AUTHOR[@]}"; do
	echo "\"$author\""                                           >> ../$builddir/build.sh
done
echo ")"                                                             >> ../$builddir/build.sh
echo "BUILD_FRAMES_DESCRIPTION=\"$BUILD_FRAMES_DESCRIPTION\""        >> ../$builddir/build.sh
echo "BUILD_FRAMES_STARTING_FRAME=\"$BUILD_FRAMES_STARTING_FRAME\""  >> ../$builddir/build.sh
echo "BUILD_FRAMES_SRCURL=\"$BUILD_FRAMES_SRCURL\""                  >> ../$builddir/build.sh
echo "BUILD_FRAMES_SRCFORMAT=\"$BUILD_FRAMES_SRCFORMAT\""            >> ../$builddir/build.sh
echo "BUILD_FRAMES_SRCSHA256=\"$BUILD_FRAMES_SRCSHA256\""            >> ../$builddir/build.sh
echo "BUILD_FRAMES_LICENSE=\"$BUILD_FRAMES_LICENSE\""                >> ../$builddir/build.sh

echo "../$builddir and its build.sh has been created."
echo "once ready, make an entry to cuts.sh"

