#!/bin/bash

ffmpeg -i $1 -filter_complex "[0:v] palettegen" /tmp/ffmpeg-palette.png
ffmpeg -i $1 -i /tmp/ffmpeg-palette.png -filter_complex "[0:v][1:v] paletteuse" $2