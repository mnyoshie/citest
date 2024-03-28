#!/bin/bash
set -e

source ./config.sh
source ../cuts.sh

pushd ..
mkdir flattened credits



# first, flatten the animation frames
for f in $(seq -f "%"${ANIMATION_BUILD_FRAMES_ZEROS}g 0 $ANIMATION_FRAMES); do
	(
	for c in ${cuts[@]}; do
		if [ -f "$c/$f.png" ]; then
			echo "$c/$f.png" >> layers-${f}.txt;
#		elif [ -f "$c/$f.jpg" ]; then
#			echo "$c/$f.jpg" >> layers-${f}.txt;
		fi
	done
	echo FLATTEN $f
	$IM_CONVERT $(cat layers-${f}.txt) -colorspace RGB -flatten flattened/$f.png
#	rm $(cat layers-${f}.txt)
	rm layers-${f}.txt
	) &

	# multithreading hack
	if [[ $(jobs -r -p | wc -l) -ge 8 ]]; then
		wait -n
	fi
done
wait

# second, build the ending credits

popd
