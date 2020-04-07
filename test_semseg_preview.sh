#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/debug:${LIBTORCH}/lib
export RUST_BACKTRACE=1

# GST_DEBUG_DUMP_DOT_DIR=dots/ \
gst-launch-1.0 \
    filesrc num-buffers=1000 location=assets/sample-04.mkv ! decodebin ! \
    aspectratiocrop aspect-ratio=10/3 ! videoscale ! videoconvert ! \
    video/x-raw,format=RGB,width=640,height=192 ! \
    semseg ! videoconvert ! fpsdisplaysink sync=false

# dot -Tpng dots/graph.dot > graph.png
