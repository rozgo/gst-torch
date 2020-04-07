#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/debug:${LIBTORCH}/lib
export RUST_BACKTRACE=1

gst-launch-1.0 --gst-disable-segtrap --gst-disable-registry-fork \
    videomixer name=comp sink_0::ypos=0 sink_1::ypos=192 sink_2::ypos=384 ! \
    x264enc pass=5 quantizer=5 ! video/x-h264, profile=high ! matroskamux ! filesink location=output.mkv sync=false \
    filesrc num-buffers=1000 location=assets/sample-04.mkv ! decodebin ! \
    aspectratiocrop aspect-ratio=10/3 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=640,height=192 ! \
    tee name=t ! \
    queue2 ! videoconvert ! comp. \
    t. ! queue2 ! monodepth ! videoconvert ! comp. \
    t. ! queue2 ! semseg ! videoconvert ! comp. 
