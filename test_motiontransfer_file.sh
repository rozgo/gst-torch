#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/release:${LIBTORCH}/lib
export RUST_BACKTRACE=1

./target/release/simbotic-stream \
    videomixer name=comp sink_0::xpos=0 sink_1::xpos=256 ! \
    x264enc pass=5 quantizer=5 ! video/x-h264, profile=high ! matroskamux ! filesink location=output.mkv sync=false \
    filesrc num-buffers=1000 location=assets/ili.mkv ! decodebin ! \
    aspectratiocrop aspect-ratio=1/1 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=256,height=256 ! \
    tee name=t ! \
        queue2 ! videoconvert ! comp. \
        t. ! queue2 ! motiontransfer source-image=${SIMBOTIC_TORCH}/assets/bigguns.png ! videoconvert ! comp.
