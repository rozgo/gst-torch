#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/release:${LIBTORCH}/lib
export RUST_BACKTRACE=1

./target/release/simbotic-stream \
    videomixer name=comp background=1 sink_1::xpos=256 sink_2::xpos=512 ! videoconvert ! xvimagesink sync=true \
    filesrc num-buffers=1000 location=assets/ili.mkv ! decodebin ! \
    tee name=t \
        t. ! queue2 ! comp. \
        facepose name=fp \
        t. ! videoconvert ! videoscale ! video/x-raw,format=BGR,width=120,height=120 ! videoconvert ! videoscale ! fp.face \
            fp.morph ! queue2 ! comp. \
            fp.landmarks ! queue2 ! comp. 