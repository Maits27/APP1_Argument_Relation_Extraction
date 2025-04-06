#!/usr/bin/bash
#SBATCH --job-name='casimedicosseq'
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-user=murruela002@ikasle.ehu.eus
#SBATCH --output=./out/secuencia_out_logs.log
#SBATCH --error=./out/secuencia_out_errors.err

#---------------------------------------------------------

source /gaueko1/users/murruela002/hf/hf.txt
export PATH="$instructenv":"$PATH"

#---------------------------------------------------------

OUTPUT_PATH='/gaueko1/users/murruela002/APP1/NLIsrc/RV/outputs_secuencia'
SRC_PATH='/gaueko1/users/murruela002/APP1/NLIsrc'
json_path="$SRC_PATH/params/train_seq_params.json"
train_script_path="$SRC_PATH/scripts/train_seq.py"
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
  porcentaje=$6
  # Hacemos X pruebas
  for i in {1..3}; do
    if [ $i -eq 1 ]; then
      seed=3000
    else
      seed=$(shuf -i 0-2999 -n 1)
    fi
    prueba="${prueba_init}_${seed}"
    
    # Copiamos la plantilla de los parámetros que hay que pasarle al script de entrenamiento
    cp "$SRC_PATH/params/plantilla_sequence_class.json" "$json_path"

    # Editamos la COPIA de la plantilla poniendo los parámetros correctos
    sed -i.bak "s#RUN_NUM#$prueba#" $json_path
    sed -i.bak "s#PORCENTAJE#$porcentaje#" $json_path
    sed -i.bak "s#END_OF_DATASET#$datos#" $json_path
    sed -i.bak "s#VAR_LR#$learning_rate#" $json_path
    sed -i.bak "s#VAR_WD#$weight_decay#" $json_path
    sed -i.bak "s#VAR_WARM#$warmup#" $json_path
    sed -i.bak "s#\"seed\": VAR_SEED#\"seed\": $seed#" $json_path

    # Lanzamos el script de entrenamiento
    python  $train_script_path

    # Limpiamos las carpetas output de los experimentos
    python "$cleaner_script_path" -i "$OUTPUT_PATH/$porcentaje/$prueba"

    # Copiamos el archivo con los parámetros del experimento a la carpeta del experimento p
    cp $json_path "$OUTPUT_PATH/$porcentaje/$prueba/params.json"

    done
}

#---------------------------------------------------------

k=0
for lr in 5e-5 75e-6
do
  for warmup in 0.1 0.05
  do
    for wd in 0.001 0.01
    do
      for data in "_sueltas"
      do
        for carpeta_porcentaje in "datos_secuencia" "data_0.05" "data_0.1" "data_0.2"
        do
          train-transformer $lr $warmup $wd $k $data $carpeta_porcentaje
          k=$((k + 1))
        done
      done
    done
  done
done


