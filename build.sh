#!/usr/bin/env bash

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=${SIMBOTIC_TORCH}/target/release:${LIBTORCH}/lib
export RUST_BACKTRACE=1

glslangValidator -V assets/shaders/facelandmark.frag -o assets/shaders/facelandmark.frag.spv && \
glslangValidator -V assets/shaders/facedepth.frag -o assets/shaders/facedepth.frag.spv && \
glslangValidator -V assets/shaders/faceskin.vert -o assets/shaders/faceskin.vert.spv && \
cargo build -j 8 --release
