import json
import logging
import evaluate
import wandb
from typing import Optional, List, Union
from pathlib import Path
import os
from datasets import load_dataset, ClassLabel#, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoConfig, PreTrainedTokenizerFast, DataCollatorWithPadding, set_seed, \
    default_data_collator
import numpy as np

# ruta_abs = ('/home/shared/esperimentuak/maitaneCasimedicos/casimedicos-NLI/src/scripts/')
# sys.path.append(ruta_abs)
# from metodos_train import get_label_list, generar_dataset

params_file_path = "/gaueko1/users/murruela002/APP1/NLIsrc/params/train_seq_params.json"
with open(params_file_path, 'r', encoding='utf-8') as f:
    script_params = json.load(f)
    print(script_params)

# -----------------------------------------------------
# SET THE SEED
set_seed(script_params['seed'])

# -----------------------------------------------------

logger = logging.getLogger(__name__)


def main():
    # Crear paths
    # if script_params['dataset_script_params['cache_dir']'] is not None and not os.path.exists(script_params['dataset_script_params['cache_dir']']):
    #     os.makedirs(script_params['dataset_script_params['cache_dir']'])

    if script_params['output_dir'] is not None and not os.path.exists(script_params['output_dir']) and script_params[
        'do_train'] and not script_params['overwrite_output_dir']:
        os.makedirs(script_params['output_dir'])
    else:
        raise ValueError(
            f"Output directory ({script_params['output_dir']}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    # Recoger ficheros entrenamiento
    data_files = {}
    if script_params['do_train']:
        if script_params['train_file'] is not None:
            data_files['train'] = script_params['train_file']
        else:
            raise ValueError('Train file error')
    print('train')
    if script_params['do_eval']:
        if script_params['dev_file'] is not None:
            data_files['dev'] = script_params['dev_file']
        else:
            raise ValueError('Dev file error')
    print('eval')
    if script_params['do_predict']:
        if script_params['test_file'] is not None:
            data_files['test'] = script_params['test_file']
        else:
            raise ValueError('Test file error')

    print('test')

    if Path(script_params['loading_script_path']).is_file():
        loading_script = script_params['loading_script_path']
    else:
        loading_script = script_params['train_file'].split(".")[-1]

    datasets = load_dataset(loading_script, data_files=data_files, cache_dir=script_params['cache_dir'])
    print('datasets')

    if script_params['do_train']:
        column_names = datasets["train"].column_names
        features = datasets["train"].features

    elif script_params['do_eval']:
        column_names = datasets["dev"].column_names
        features = datasets["dev"].features

    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features
    label_column_name = (
        f"{script_params['task_name']}_tags" if f"{script_params['task_name']}_tags" in column_names else column_names[
            -1]
    )
    # label_list = ['entailment', 'neutral', 'contradiction']

    label_to_id = {
        "no-relation": 0,
        "attack": 1,
        "support": 2
    }
    label_column_name = "label"
    # Convertir las etiquetas del conjunto de datos a IDs utilizando label_to_id
    if script_params['do_train'] and label_column_name in datasets["train"].features:
        datasets["train"] = datasets["train"].map(lambda example: {"label": label_to_id[example[label_column_name]]})
    if script_params['do_eval'] and label_column_name in datasets["dev"].features:
        datasets["dev"] = datasets["dev"].map(lambda example: {"label": label_to_id[example[label_column_name]]})
    if script_params['do_predict'] and label_column_name in datasets["test"].features:
        datasets["test"] = datasets["test"].map(lambda example: {"label": label_to_id[example[label_column_name]]})

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Cargamos la configuración del modelo:
    config = AutoConfig.from_pretrained(
        script_params['config_name'] if script_params['config_name'] else script_params['model_path'],
        num_labels=3,
        finetuning_task=script_params['task_name'],
        cache_dir=script_params['cache_dir'],
    )
    # Cargamos el modelo:
    model = AutoModelForSequenceClassification.from_pretrained(
        script_params['model_path'],
        config=config,
        cache_dir=script_params['cache_dir'],
    )
    print('tokenizer')

    # Cargamos el tokenizador:
    tokenizer = AutoTokenizer.from_pretrained(
        script_params['tokenizer_name'] if script_params['tokenizer_name'] else script_params['model_path'],
        cache_dir=script_params['cache_dir'],
        use_fast=True,
        config=config,
        add_prefix_space=True,
        label_column_name=label_column_name
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )
    print('tokenize')

    # Tokenizar hypothesis y premisa
    def tokenizar_textos(textos):
        args = (textos['premise'], textos['hypothesis'])
        return tokenizer(
            *args,
            padding='max_length',
            truncation="only_first",
            return_tensors='pt',
        )

    print(f"overwrite cache: {script_params['overwrite_cache']}")
    tokenized_datasets = datasets.map(
        tokenizar_textos,
        batched=True,
        # num_proc=script_params['preprocessing_num_workers'],
        load_from_cache_file=not script_params['overwrite_cache']
    )
    print('collator')
    # Data collator TODO no se si es necesario en mi caso
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Set evaluetaion function to be used during training (accuracy, f1, ...)
    accuracy = evaluate.load('accuracy')  # Load the accuracy function
    f1 = evaluate.load('f1')  # Load the f-score function
    precision = evaluate.load('precision')  # Load the precision function
    recall = evaluate.load('recall')  # Load the recall function
    cnf_matrix = evaluate.load('BucketHeadP65/confusion_matrix')  # Load the confusion matrix function



    def compute_metrics(eval_pred):
        # class_names = ["entailment", "neutral", "contradiction"]
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy_value = accuracy.compute(predictions=predictions, references=labels)

        f1_value_micro = f1.compute(predictions=predictions, references=labels, average='micro')
        f1_value_macro = f1.compute(predictions=predictions, references=labels, average='macro')
        f1_value_weighted = f1.compute(predictions=predictions, references=labels, average='weighted')
        f_conj = f1.compute(predictions=predictions, references=labels, labels=[1, 2], average='macro')
        class_f1 = f1.compute(predictions=predictions, references=labels, average=None)

        precision_value_micro = precision.compute(predictions=predictions, references=labels, average='micro')
        precision_value_macro = precision.compute(predictions=predictions, references=labels, average='macro')
        precision_value_weighted = precision.compute(predictions=predictions, references=labels, average='weighted')
        class_precision = precision.compute(predictions=predictions, references=labels, average=None,
                                            zero_division='warn')

        recall_value_micro = recall.compute(predictions=predictions, references=labels, average='micro')
        recall_value_macro = recall.compute(predictions=predictions, references=labels,  average='macro')
        recall_value_weighted = recall.compute(predictions=predictions, references=labels, average='weighted')
        class_recall = recall.compute(predictions=predictions, references=labels, average=None,
                                      zero_division='warn')

        confusion_matrix_serializable = cnf_matrix.compute(predictions=predictions, references=labels)

        #  Every element in the return dict, must be serializable so we log confusion_matrix (which is a plot object from wanbd library) independently to wandb
        confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions,
                                                       class_names=["no-relation", "attack", "support"])
        wandb.log({'confusion_matrix': confusion_matrix})

        return {
            'accuracy': accuracy_value['accuracy'],
            'f1_micro': f1_value_micro['f1'],
            'f1_macro': f1_value_macro['f1'],
            'f1_weighted': f1_value_weighted['f1'],
            'f1_norel': class_f1['f1'][0],
            'f1_att': class_f1['f1'][1],
            'f1_supp': class_f1['f1'][2],
            "f1_a_s": f_conj['f1'],
            'precision_micro': precision_value_micro['precision'],
            'precision_macro': precision_value_macro['precision'],
            'precision_weighted': precision_value_weighted['precision'],
            'precision_norel': class_precision['precision'][0],
            'precision_att': class_precision['precision'][1],
            'precision_supp': class_precision['precision'][2],
            'recall_micro': recall_value_micro['recall'],
            'recall_macro': recall_value_macro['recall'],
            'recall_weighted': recall_value_weighted['recall'],
            'recall_norel': class_recall['recall'][0],
            'recall_att': class_recall['recall'][1],
            'recall_supp': class_recall['recall'][2],
            'confusion_matrix': confusion_matrix_serializable['confusion_matrix'].tolist()
        }

    training_args = TrainingArguments(
        output_dir=script_params['output_dir'],
        num_train_epochs=script_params['num_train_epochs'],  # número de épocas de entrenamiento
        per_device_train_batch_size=script_params['per_device_train_batch_size'],
        # tamaño del lote de entrenamiento por dispositivo
        per_device_eval_batch_size=script_params['per_device_eval_batch_size'],
        gradient_accumulation_steps=script_params['gradient_accumulation_steps'],
        learning_rate=script_params['learning_rate'],
        seed=script_params['seed'],
        weight_decay=script_params['weight_decay'],
        warmup_ratio=script_params['warmup_ratio'],
        evaluation_strategy=script_params['evaluation_strategy'],
        save_strategy=script_params['save_strategy'],
        save_total_limit=2,
        metric_for_best_model=script_params['metric_for_best_model'],
        report_to=script_params['report_to'],
        run_name=script_params['run_name'],
        # logging_dir='./logs',            # directorio donde se guardarán los registros de entrenamiento
        # logging_steps=10,
        load_best_model_at_end=script_params['load_best_model_at_end'],
        # use_cpu=True
    )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!! TRAINING ARGUMENTS !!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(training_args)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if script_params['do_train'] else None,
        eval_dataset=tokenized_datasets["dev"] if script_params['do_eval'] else None,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[],
    )

    def predict_and_save(prediction_dataset: str, output_file: str, metric_key_prefix: str = 'eval'):
        if trainer.is_world_process_zero():
            prediction_dataset = tokenized_datasets[prediction_dataset]
            output_predictions_file = os.path.join(script_params['output_dir'], output_file)

            prediction_results = trainer.evaluate(prediction_dataset)

            # Log evaluation
            logger.info("***** Eval results *****")
            for key, value in prediction_results.items():
                logger.info(f"  {key} = {value}")

            # Save evaluation in json
            with open(f'{output_predictions_file}_results.json', "w", encoding='utf8') as writer:
                json.dump(prediction_results, writer, ensure_ascii=False, indent=2)

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_dir = os.path.join(script_params['output_dir'], "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_dir, "w", encoding='utf8') as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    with open(f'{script_params["output_dir"]}/params.json', 'w', encoding='utf8') as archivo_json:
        json.dump(script_params, archivo_json, ensure_ascii=False, indent=4)

    results = {}
    if script_params['do_eval'] == True:
        logger.info("*** Evaluate ***")
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EVALUATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        predict_and_save(prediction_dataset='dev', output_file='eval_predictions')

    # Predict
    if script_params['do_predict'] == True:
        logger.info("*** Predict ***")

        predict_and_save(prediction_dataset='test', output_file='test_predictions')

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
