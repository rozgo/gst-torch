#!/bin/bash

ffmpeg -i $1.mkv -pix_fmt yuv420p -c:v h264_nvenc -preset slow -cq 10 -bf 2 -g 150 $2.mp4
