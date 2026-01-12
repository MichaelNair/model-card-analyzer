---
datasets:
- EvaKlimentova/knots_AF
metrics:
- accuracy
---

# Knots ProtBert-BFD AlphaFold

Fine-tuned [ProtBert-BFD](https://huggingface.co/Rostlab/prot_bert_bfd) to classify proteins as knotted vs. unknotted. 

## Model Details

- **Model type:** Bert
- **Language:** proteins (amino acid sequences)
- **Finetuned from model:** [Rostlab/prot_bert_bfd](https://huggingface.co/Rostlab/prot_bert_bfd)

Model Sources:

- **Repository:** [CEITEC](https://github.com/ML-Bioinfo-CEITEC/pknots_experiments)
- **Paper:** TBD

## Usage

Dataset format:
```
id,sequence,label
A0A2W5F4Z7,MGGIFRVNTYYTDLEPYLQSTKLPIYGALLDGENIYELVDKSKGILVIGNESKGIRSTIQNFIQKPITIPRIGQAESLNAAVATGIIVGQLTL,1
...
```

Load the dataset:
```
import pandas as pd
from datasets import Dataset, load_dataset

df = pd.read_csv(INPUT, sep=',')
dss = Dataset.from_pandas(df)
```

Predict:
```
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from math import exp

def tokenize_function(s):
    seq_split = ' '.join(s['Sequence'])
    return tokenizerM1(seq_split)

tokenizer = AutoTokenizer.from_pretrained('roa7n/knots_protbertBFD_alphafold')
model = AutoModelForSequenceClassification.from_pretrained('roa7n/knots_protbertBFD_alphafold')

tokenized_dataset = dss.map(tokenize_function, num_proc=4)
tokenized_dataset.set_format('pt')
tokenized_dataset

training_args = TrainingArguments(<PATH>, fp16=True, per_device_eval_batch_size=50, report_to='none')  

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizerM1
)

predictions, _, _ = trainer.predict(tokenized_dataset)
predictions = [np.exp(p[1]) / np.sum(np.exp(p), axis=0) for p in predictions]
df['preds'] = predictions
```

## Evaluation

Per protein family metrics:

|       M1 ProtBert-BFD        | Dataset size | Unknotted set size | Accuracy |   TPR  |   TNR  |
|:----------------------------:|:------------:|:------------------:|:--------:|:------:|:------:|
| All                          | 39412        | 19718              | **0.9845**   | 0.9865 | 0.9825 |
| SPOUT                        | 7371         | 550                | 0.9887   | 0.9951 | 0.9090 |
| TDD                          | 612          | 24                 | 0.9901   | 0.9965 | 0.8333 |
| DUF                          | 716          | 429                | 0.9748   | 0.9721 | 0.9766 |
| AdoMet synthase              | 1794         | 240                | 0.9899   | 0.9929 | 0.9708 |
| Carbonic anhydrase           | 1531         | 539                | 0.9588   | 0.9737 | 0.9313 |
| UCH                          | 477          | 125                | 0.9056   | 0.9602 | 0.7520 |
| ATCase/OTCase                | 3799         | 3352               | 0.9994   | 0.9977 | 0.9997 |
| ribosomal-mitochondrial      | 147          | 41                 | 0.8571   | 1.0000 | 0.4878 |
| membrane                     | 8225         | 1493               | 0.9811   | 0.9904 | 0.9390 |
| VIT                          | 14262        | 12555              | 0.9872   | 0.9420 | 0.9933 |
| biosynthesis of lantibiotics | 392          | 286                | 0.9642   | 0.9528 | 0.9685 |


## Citation [optional]

**BibTeX:** TODO

## Model Authors

Simecek: simecek@mail.muni.cz
Klimentova: vae@mail.muni.cz
Sramkova: denisa.sramkova@mail.muni.cz