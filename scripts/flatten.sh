#!/bin/bash
set -ex

source ../cuts.sh
source ./config.sh

pushd ..
mkdir flattened
for f in $(seq -f "%04g" 0 $frames); do
	for c in ${cuts[@]}; do
		if [ -f "$c/$f.png" ]; then
			echo "$c/$f.png" >> layers.txt;
		fi
	done
	convert `cat layers.txt` -flatten flattened/$f.png
	rm layers.txt
done
popd
	
