#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/debug:${LIBTORCH}/lib
export RUST_BACKTRACE=1

gst-launch-1.0 \
    videomixer name=comp sink_0::xpos=0 sink_1::xpos=256 ! \
    xvimagesink sync=false \
    v4l2src ! \
    aspectratiocrop aspect-ratio=1/1 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=256,height=256 ! \
    tee name=t ! \
    queue2 ! videoconvert ! comp. \
    t. ! queue2 ! motiontransfer ! videoconvert ! comp.
