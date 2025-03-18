#!/bin/bash

#PYTHONPATH=/home/shared/esperimentuak/maitaneCasimedicos/a2t/Ask2Transformers
#export PYTHONPATH
#export PYTHONPATH=/home/shared/esperimentuak/maitaneCasimedicos/a2t/Ask2Transformers:$PYTHONPATH
#echo $PYTHONPATH

SRC_PATH='/home/shared/esperimentuak/maitaneCasimedicos/src'
#dev_script_path="$SRC_PATH/Inferencia/a2t.py"
dev_script_path="/home/shared/esperimentuak/maitaneCasimedicos/a2t/Ask2Transformers/0Shot_a2t_script.py"
json_path="$SRC_PATH/params/0shot_params.json"

function a2t_run(){
    dataset=$1
    thres=0.09309309309309309
    echo $thres
    multzo=$2
#    echo Dataset $dataset
    OUTPUT_PATH="/home/shared/esperimentuak/maitaneCasimedicos/src/Inferencia/output_0shot_${multzo}"

    cp "$SRC_PATH/params/0Shot_a2t_plantilla.json" "$json_path"

    sed -i.bak "s#NOMBRE_DATASET#$dataset#" $json_path
    sed -i.bak "s#MULTZO#$multzo#" $json_path
    sed -i.bak "s#THRESHOLD#$thres#" $json_path

    python $dev_script_path

    cp $json_path "$OUTPUT_PATH/params.json"
}

for data in 0
do
    for grupo in "test"
    do
        a2t_run $data $grupo
    done
done
