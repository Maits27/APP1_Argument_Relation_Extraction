#!/usr/bin/env bash
#SBATCH --job-name='casimedicosNLI_a2t'
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=murruela002@ikasle.ehu.eus
#SBATCH --output=./out/a2t_out_logs.log
#SBATCH --error=./out/a2t_out_errors.err

source /gaueko1/users/murruela002/hf/hf.txt
# Behar dugun igurune birtuala aktibatzen dugu
export PATH="$instructenv":"$PATH"

SRC_PATH='/gaueko1/users/murruela002/APP1/NLIsrc'
#dev_script_path="$SRC_PATH/Inferencia/a2t.py"
dev_script_path="/gaueko1/users/murruela002/APP1/NLIsrc/Ask2Transformers/a2t_script.py"
json_path="$SRC_PATH/params/inferencia_params.json"

function a2t_run(){
    dataset=$1
    thres=0.8

    echo $thres
    multzo=$2
#    echo Dataset $dataset
    OUTPUT_PATH="/gaueko1/users/murruela002/APP1/NLIsrc/Inferencia/outputs/output${dataset}_${multzo}"

    cp "$SRC_PATH/params/a2t_plantilla.json" "$json_path"

    sed -i.bak "s#NOMBRE_DATASET#$dataset#" $json_path
    sed -i.bak "s#MULTZO#$multzo#" $json_path
    sed -i.bak "s#THRESHOLD#$thres#" $json_path

    python $dev_script_path

    cp $json_path "$OUTPUT_PATH/params.json"
}

for data in 1
do
    for grupo in "test"
    do
        a2t_run $data $grupo
    done
done


if [[ $dataset == 1 ]]; then
   thres=0.995995995995996
fi
if [[ $dataset == 2 ]]; then
   thres=0.7507507507507507
fi
if [[ $dataset == 3 ]]; then
   thres=0.35235235235235235
fi
if [[ $dataset == 4 ]]; then
   thres=0.996996996996997
fi
if [[ $dataset == 5 ]]; then
   thres=1.0
fi
if [[ $dataset == 6 ]]; then
   thres=1.0
fi
if [[ $dataset == 7 ]]; then
   thres=0.997997997997998
fi
if [[ $dataset == 8 ]]; then
   thres=0.998998998998999
fi
if [[ $dataset == 9 ]]; then
   thres=1.0
fi
if [[ $dataset == 10 ]]; then
   thres=0.997997997997998
fi
if [[ $dataset == 11 ]]; then
   thres=1.0
fi