#!/usr/bin/env bash
#SBATCH --job-name='casimedicosNLI_a2t'
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=murruela002@ikasle.ehu.eus
#SBATCH --output=./out/a2t_out_logs.log
#SBATCH --error=./out/a2t_out_errors.err

source /gaueko1/users/murruela002/hf/hf.txt
# Behar dugun igurune birtuala aktibatzen dugu
export PATH="$instructenv":"$PATH"

SRC_PATH='/gaueko1/users/murruela002/APP1/NLIsrc'
dev_script_path="/gaueko1/users/murruela002/APP1/NLIsrc/Ask2Transformers/a2t_script.py"
json_path="$SRC_PATH/params/inferencia_params.json"

function a2t_run(){
    dataset=$1
    thres=0.2

    echo $thres
    multzo=$2
    OUTPUT_PATH="/gaueko1/users/murruela002/APP1/NLIsrc/Inferencia/outputs/output${dataset}_${multzo}"

    cp "$SRC_PATH/params/a2t_plantilla.json" "$json_path"

    sed -i.bak "s#NOMBRE_DATASET#$dataset#" $json_path
    sed -i.bak "s#MULTZO#$multzo#" $json_path
    sed -i.bak "s#THRESHOLD#$thres#" $json_path
    echo entro
    python $dev_script_path

    cp $json_path "$OUTPUT_PATH/params.json"
}

for data in 0.05 0.1 0.2 100
do
    for grupo in "dev"
    do
        a2t_run $data $grupo
    done
done
