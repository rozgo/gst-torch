#!/bin/bash

echo " video_in: $1"
echo "    width: $2"
echo "   height: $3"
echo "video_out: $4"
ffmpeg -i $1.mkv -an -vf scale=$2:$3 $4.mkv
# -filter:v "crop=1280:520:0:200"