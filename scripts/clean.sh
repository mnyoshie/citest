#!/bin/bash
set -ev

pushd ..
find -name \*.png -exec rm {} \;
find -name \*.zip -exec rm {} \;
rm -rf flattened
popd
