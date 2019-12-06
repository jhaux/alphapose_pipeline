# export ROOT=/export/scratch/jhaux/Data/olympic\ sports/
# export ROOT=/export/scratch/jhaux/Data/olympic_test/

export CUDA_VISIBLE_DEVICES=$2


function framer() {
    file=$1
    echo "make_frames ${file}"
    mkdir "${file}_frames"

    # if [ ${file: -4} != ".mp4" ]; then
    #     # If the file is in some strange format (not mp4) convert it to mp4
    #     # which is easy to use for me, e.g. for preview.
    #     ffmpeg -y -i "${file}" "${file}.mp4" -hide_banner
    # fi

    # Extract frames from the original video to avoid compression loss
    ffmpeg -y -i "${file}" -vf yadif "${file}_frames/frame_%04d.png" -hide_banner
}

framer "$1"

# OIFS="$IFS"
# IFS=$'\n'
# echo '========' &>> frame_log.txt
# for file in `find $ROOT -name "*.seq" | sort`  
# do
#         (framer $file &>> frame_log.txt && echo "") &
# done  |  tqdm --total $(find "$ROOT" -name *.seq | wc -l)
# IFS="$OIFS"
