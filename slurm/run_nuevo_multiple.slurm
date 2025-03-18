#!/usr/bin/env bash
#SBATCH --job-name='casimedicosNLI'
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=murruela002@ikasle.ehu.eus
#SBATCH --output=./out/out_logs.log
#SBATCH --error=./out/out_errors.err

#---------------------------------------------------------
# Preparar el entorno

# source /var/python3envs/transformers-4.12.3/bin/activate
source /gaueko1/users/murruela002/hf/hf.txt
# Behar dugun igurune birtuala aktibatzen dugu
export PATH="$instructenv":"$PATH"
# export HF_HOME=/gscratch/users/USER/MY_TRANSFORMER_evaluation
# export TRANSFORMERS_CACHE=/gscratch/users/USER/MY_TRANSFORMER_evaluation/transformers

#---------------------------------------------------------
# Definir paths (a ser posible que sean absolutos)
# export CUDA_LAUNCH_BLOCKING=1
# unset CUDA_VISIBLE_DEVICES
OUTPUT_PATH='/gaueko1/users/murruela002/APP1/NLIsrc/output'

SRC_PATH='/gaueko1/users/murruela002/APP1/NLIsrc'
json_path="$SRC_PATH/params/train_params.json"
train_script_path="$SRC_PATH/scripts/training.py"
cleaner_script_path="$SRC_PATH/scripts/utils/experiment_cleaner.py"

#---------------------------------------------------------

# Definimos la función encargada de hacer los experimentos
function train-transformer() {
  # Obtenemos el learning rate que se obtendrá por parámetro
  learning_rate=$1
  weight_decay=$3
  warmup=$2
  prueba_init=$4
  datos=$5
  # Hacemos X pruebas
  for i in {1..1}; do
    if [ $i -eq 1 ]; then
      seed=3000
    else
      seed=$(shuf -i 0-2999 -n 1)
    fi
    prueba="${prueba_init}"
    
    # Copiamos la plantilla de los parámetros que hay que pasarle al script de entrenamiento
    cp "$SRC_PATH/params/plantilla.json" "$json_path"
#    cp "$SRC_PATH/scripts/training.py" "$OUTPUT_PATH"

    # Generamos una seed random
    # seed=3000

    # Editamos la COPIA de la plantilla poniendo los parámetros correctos
    sed -i.bak "s#RUN_NUM#$prueba#" $json_path
    sed -i.bak "s#END_OF_DATASET#$datos#" $json_path
    sed -i.bak "s#VAR_LR#$learning_rate#" $json_path
    sed -i.bak "s#VAR_WD#$weight_decay#" $json_path
    sed -i.bak "s#VAR_WARM#$warmup#" $json_path
    sed -i.bak "s#\"seed\": VAR_SEED#\"seed\": $seed#" $json_path

    # Lanzamos el script de entrenamiento
    python  $train_script_path

    # Limpiamos las carpetas output de los experimentos
    python "$cleaner_script_path" -i "$OUTPUT_PATH/$prueba"

    # Copiamos el archivo con los parámetros del experimento a la carpeta del experimento p
    cp $json_path "$OUTPUT_PATH/$prueba/params.json"
    cp "$SRC_PATH/scripts/training.py" "$OUTPUT_PATH"
    #cp $json_path "$OUTPUT_PATH/$prueba"
    done
    python "/gaueko1/users/murruela002/APP1/NLIsrc/scripts/csv_outputs.py"
    #python "/gaueko1/users/murruela002/APP1/NLIsrc/scripts/matriz.py"
}

#---------------------------------------------------------
# Definir que experimentos hacer con qué learning rates (referencia: el por defecto de Hugg>

#train-transformer "75e-6" "0.05" "0.01" "0" ""
#train-transformer "75e-6" "0.05" "0.01" "1" "_evitando_concatenaciones"
#train-transformer "75e-6" "0.0" "0.01" "2" "_evitando_concatenaciones_viejo"
#train-transformer "5e-5" "0.05" "0.01" "3" "_sueltas"
# for data in "" "_evitando_concatenaciones_viejo" "_evitando_concatenaciones" "_sueltas"

k=0
for lr in 5e-5
do
  for warmup in 0.05
  do
    for wd in 0.01
    do
      for data in "_sueltas"
      do
        train-transformer $lr $warmup $wd $k $data
        k=$((k + 1))
      done
    done
  done
done


