VIDEO=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=$GPU

function maskinator() {
    file=$1
    echo $file
    python estimate_masks.py --video "${file}"
}

maskinator "$VIDEO"
