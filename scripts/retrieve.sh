#!/bin/bash
set -ev

source ../cuts.sh

for c in ${cuts[@]}; do
	pushd ../$c
	curl -L $(head -1 < properties.txt | awk '{printf $2}') -o $c.zip
	unzip $c.zip
	popd
done
