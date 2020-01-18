#!/bin/bash

echo "  video_in: $1"
echo "start_time: $2"
echo "  duration: $3"
echo " video_out: $4"
ffmpeg -i $1.mkv -ss $2 -t $3 -an -c:v copy $4.mkv
