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
        results_json_path = (params_json_path.parent / 'eval_predictions_results.json')

        with open(params_json_path, 'r', encoding='utf-8') as file:
            params = json.load(file)

        with open(results_json_path, 'r', encoding='utf-8') as file:
            datos_json = json.load(file)

        if 'warmup_ratio' in params: wr = params['warmup_ratio']
        else: wr = 0.0

        datos_json.update({
            "LR": params['learning_rate'],
            "WD": params['weight_decay'],
            'seed': params['seed'],
            'warmup': wr,
            'batch_size': params['per_device_train_batch_size'],
            'gradient_accumulation': params['gradient_accumulation_steps'],
            "percentage": params['train_file'].split('/')[-2],
            'train_files': params['train_file'].split('/')[-1],
            "Name_of_experiment": params['run_name']
        })

        datos_combinados.append(datos_json)

        with open(f'{initPath}/todasLasMetricas_dev.json', 'w', encoding='utf-8') as file:
            json.dump(datos_combinados, file, indent=2)

    return json_to_dataframe(f'{initPath}/todasLasMetricas_dev.json')


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
    sacarCSV('/gaueko1/users/murruela002/APP1/NLIsrc/outputs', 'output_dev.csv', 'output_dev.xlsx')
