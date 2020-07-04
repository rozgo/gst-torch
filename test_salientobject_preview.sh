#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/release:${LIBTORCH}/lib
export RUST_BACKTRACE=1

./target/release/simbotic-stream \
    filesrc num-buffers=1000 location=assets/joker.mkv ! decodebin ! \
    aspectratiocrop aspect-ratio=1/1 ! videoscale ! \
    videoconvert ! videorate ! video/x-raw,format=RGB,width=320,height=320,framerate=0/1 ! \
    tee name=rgb \
    rgb. ! queue2 ! videoconvert ! comp.sink_0 \
    rgb. ! queue2 ! videoconvert ! blend.sink_0 \
    rgb. ! queue2 ! salientobject ! tee name=mask \
    mask. ! queue2 ! videoconvert ! comp.sink_1 \
    mask. ! queue2 ! videoconvert ! blend.sink_1 \
    frei0r-mixer-multiply name=blend ! videoconvert ! videoscale ! video/x-raw,width=640,height=640 ! xvimagesink sync=false \
    compositor name=comp sink_0::xpos=0 sink_1::xpos=320 ! videoconvert ! xvimagesink sync=false

