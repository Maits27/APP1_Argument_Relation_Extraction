import json

def transform_file(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    datos_transformados = []
    for instancia in datos:

        if instancia["label"] == "contradiction":
            continue
        h = instancia["hypothesis"].split('" support \"')
        label = "support"
        if len(h)==1:
            h = instancia["hypothesis"].split('" attack \"')
            label = "attack"
        datos_transformados.append({
            "premise": instancia["premise"],
            "hypothesis": f"{h[0]}</s>{h[1]}",
            "label": "no-relation" if instancia["label"] == "neutral" else label
        })
    with open(outfile, "w", encoding='utf-8') as f:
        json.dump(datos_transformados, f, ensure_ascii=False, indent=4)


# for i in ["train_sueltas", "dev", "test"]:
#     infile = f"/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/{i}.json"
#     outfile = f"/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/datos_secuencia/sequence_{i}.json"
#
#     transform_file(infile, outfile)


for folder in ["data_0.05", "data_0.1", "data_0.2"]:
    infile = f"/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/{folder}/train_sueltas.json"
    outfile = f"/gaueko1/users/murruela002/APP1/NLIsrc/Datasets/{folder}/sequence_train_sueltas.json"

    transform_file(infile, outfile)

