#!/bin/bash
set -ev

pushd ..
rm -rf flattened credits
rm */*.png
rm layers-*.txt
popd
