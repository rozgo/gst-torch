#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_GSTTORCH}/target/debug:${LIBTORCH}/lib
export RUST_BACKTRACE=1

cargo build -j 8
