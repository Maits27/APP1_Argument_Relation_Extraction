# APP1 Final Project: Argumentative AI for Medical Diagnosis Support

This work presents an approach for enhancing medical diagnosis support by extracting argumentative relationships form clinical texts. 

We compare a two-step Natural Language Inference (NLI) method with a conventional one-step text classification approach, by identifying \textit{Support} and \textit{Attack} relations. 

## Used external tools and models
As stated before, two models will be used to train the different methods to be compared. These are:

* For text clssification methodology: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)
* For NLI-based methodology: [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)

Also, for the NLI-based methodology, we leverage [Ask2Transformers](https://github.com/osainz59/Ask2Transformers/tree/master) library in order to perform multiple zero-shot NLI trials with varied verbalizations per relation and predicting the most probable relationship.

For that purpose we clone the Ask2Transformers repository and execute [a2t_script.py](Inferencia/a2t_script.py) script that we will locate in the root folder of Ask2Transformers. 
