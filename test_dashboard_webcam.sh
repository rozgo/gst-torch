#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/release:${LIBTORCH}/lib
export RUST_BACKTRACE=1

./target/release/simbotic-stream  \
    videomixer name=comp sink_0::ypos=0 sink_1::ypos=192 sink_2::ypos=384 ! \
    xvimagesink sync=false \
    v4l2src ! \
    aspectratiocrop aspect-ratio=10/3 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=640,height=192 ! \
    tee name=t ! \
    queue2 ! videoconvert ! comp. \
    t. ! queue2 ! monodepth ! videoconvert ! comp. \
    t. ! queue2 ! semseg ! videoconvert ! comp. 
