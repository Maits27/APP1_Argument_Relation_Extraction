# APP1 Final Project: Argumentative AI for Medical Diagnosis Support

This work presents an approach for enhancing medical diagnosis support by extracting argumentative relationships form clinical texts. 

We compare a two-step Natural Language Inference (NLI) method with a conventional one-step text classification approach, by identifying \textit{Support} and \textit{Attack} relations. 

## Used external tools and models
As stated before, two models will be used to train the different methods to be compared. These are:

* For text clssification methodology: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base).
* For NLI-based methodology: [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).

Also, for the NLI-based methodology, we leverage [Ask2Transformers](https://github.com/osainz59/Ask2Transformers/tree/master) library in order to perform multiple zero-shot NLI trials with varied verbalizations per relation and predicting the most probable relationship.

For that purpose we clone the Ask2Transformers repository and execute [a2t_script.py](Inferencia/a2t_script.py) script that we will locate in the root folder of Ask2Transformers. 

## Tools in this repository

As for the elememts in this repository:

### DATASETS
[Datasets](Datasets) folder contains the data needed for the training and testing of the model for both methodologies. All dolders containing ``secuencia'' word in it corresponds to the text classidication methodology and the rest to the NLI methodology. The [data_0.05](Datasets/data_0.05), [data_0.1](Datasets/data_0.1) and [data_0.2](Datasets/data_0.2) folders contain the training files for the different simulations of data-scarce environments.

### PARAMS
[params](params) folder contains the templates of the different experiments with their corresponding configurations. The missing values or the ones that are in higher case will be replaced using the slurm folder's executables.

### SCRIPTS
[scripts](scripts) folder contains all the scripts needed to train the two different models (```metodos_train.py```, ```train_seq.py``` and ```training.py```), the ones to visualize the results (```csv_outputs.py``` and ```matriz.py```) and two more folders: 

* [preprocess](scripts/preprocess) to preprocess the data of casiMedicos and convert it to the different training files.
* [utils](scripts/utils) with the scripts used to clean the output files and other helping methods.

### SLURM
[slurm](slurm) folder contains the bash files needed to change all the hyperparameters located in params folder and execute the training script with the different configurations for all the experiments. [run_secuencias](slurm/run_secuencias.slurm) corresponds to the experiments of text classification, [run_nli_train](slurm/run_nli_train.slurm) to the NLI models' training and the [run_a2t](slurm/run_a2t.slurm) bash is executed once we have the trained NLI model to perform the multiple zero-shots using Ask2Transformers to predict the relations.

### INFERENCIA
[Inferencia](Inferencia) folder contains the data needed to evaluate the NLI model in the corresponding format ([Datasets](Inferencia/Datasets)) and the script to be placed in Ask2Transformers repository.

## RESULTS
The results for the different experiments are located in the following folders:
* Text classification results: [Results_models/Text-Classification](Results_models/Text-Classification)
* NLI model training results: [Results_models/NLI](Results_models/NLI)
* Ask2Transformers results leveraging the NLI models: [Inferencia/outputs](Inferencia/outputs)

