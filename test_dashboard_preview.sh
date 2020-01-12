#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=`pwd`/target/debug:${LIBTORCH}/lib
export RUST_BACKTRACE=1

gst-launch-1.0 --gst-disable-segtrap --gst-disable-registry-fork \
    videomixer name=comp sink_0::ypos=0 sink_1::ypos=192 ! \
    xvimagesink sync=false \
    filesrc num-buffers=1000 location=assets/sample-01.webm ! decodebin ! \
    aspectratiocrop aspect-ratio=10/3 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=640,height=192 ! \
    tee name=t ! \
    queue2 ! monodepth ! videoconvert ! comp. \
    t. ! queue2 ! videoconvert ! comp.
