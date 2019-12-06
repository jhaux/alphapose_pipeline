# export ROOT=/export/scratch/jhaux/Data/olympic\ sports/
# export ROOT=/export/scratch/jhaux/Data/olympic_test/

export CUDA_VISIBLE_DEVICES=$2
export MPLBACKEND=AGG

function tracker () {
    file=$1
    cd PoseFlow
    echo $file; python tracker-general.py \
            --imgdir "${file}_frames/" \
            --in_json "${file}_track/alphapose-results.json" \
            --out_json "${file}_track/alphapose-forvis-tracked.json"  # \
            # --visdir "${file}_track/vis"
    # ffmpeg -y -i "${file}_track/vis/frame_%04d.png.png" "${file}_track/vis/vis.mp4" -hide_banner
    # ffmpeg -y -f image2 -i "${file}_track/vis/frame_%*.png" -vcodec libx264 -preset faster "${file}_track/vis/vis.mp4" -hide_banner
    # if [[ -f "${file}_track/vis/vis.mp4" ]]; then
    #         rm ${file}_track/vis/*.png
    # fi
}

tracker "$1"

# OIFS="$IFS"
# IFS=$'\n'
# echo '======' &>> track_log.txt
# for file in `find $ROOT -name "*.seq" | sort`  
# do
#     (tracker &>> track_log.txt && echo '')
# done  |  tqdm --total $(find "$ROOT" -name *.seq | wc -l)
# # done  |  tqdm --total $(( 2 + $(( 10 * $(find "$ROOT" -name *.seq | wc -l) )) ))
# IFS="$OIFS"
