import json
from pathlib import Path
import openpyxl
import pandas as pd

def json_to_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        return pd.json_normalize(data)

def combinarJSON(init):
    datos_combinados = []
    initPath = Path(init)

    for params_json_path in list(initPath.glob('**/params.json')):
        results_json_path = (params_json_path.parent / 'results.json')
        if str(params_json_path.parent).split('_')[-1] == 'dev':

            with open(params_json_path, 'r', encoding='utf-8') as file:
                params = json.load(file)

            with open(results_json_path, 'r', encoding='utf-8') as file:
                datos_json = json.load(file)

            modelo_resultados_path = (f'{params["model_path"]}/eval_predictions_results.json')
            modelo_params_path = (f'{params["model_path"]}/params.json')

            with open(modelo_resultados_path, 'r', encoding='utf-8') as file:
                modelo_resultados = json.load(file)

            with open(modelo_params_path, 'r', encoding='utf-8') as file:
                modelo_params = json.load(file)

            if 'warmup_ratio' in modelo_params: wr = modelo_params['warmup_ratio']
            else: wr = 0.0


            datos_json.update({
                'f1_ent_modelo': modelo_resultados['eval_f1_ent'],
                'init_threshold':params['negative_threshold'],
                'modelo': {
                    "LR": modelo_params['learning_rate'],
                    "WD": modelo_params['weight_decay'],
                    'seed': modelo_params['seed'],
                    'warmup': wr,
                    "epochs": modelo_params['num_train_epochs'],
                    'batch_size': modelo_params['per_device_train_batch_size'],
                    'gradient_accumulation': modelo_params['gradient_accumulation_steps'],
                    'train_files': modelo_params['train_file'].split('/')[-1]
                },
                'templates':params['templates'],
                'model_path': params['model_path'],
                'data_path': params['data_path'],
                'output_dir': params['output_dir']
            })

            datos_combinados.append(datos_json)

            with open(f'{initPath}/resultadosFinales.json', 'w', encoding='utf-8') as file:
                json.dump(datos_combinados, file, indent=2)

    return json_to_dataframe(f'{initPath}/resultadosFinales_test.json')


def sacarCSV(initPath, output_csv, output_excel):
    all_dataframes = combinarJSON(initPath)
    all_dataframes.to_csv(f'{initPath}/{output_csv}', index=False)
    csv_a_excel(initPath, output_csv, output_excel)

def csv_a_excel(initPath, output_csv, output_excel):
    # Lee el archivo CSV
    df = pd.read_csv(f'{initPath}/{output_csv}')

    # Guarda el DataFrame en un archivo de Excel
    df.to_excel(f'{initPath}/{output_excel}', index=False)


if __name__ == "__main__":
    sacarCSV('/home/shared/esperimentuak/maitaneCasimedicos/src/Inferencia/outputs', 'output.csv', 'output.xlsx')