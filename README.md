# **Artifact for the Paper "Properly Offer Options to Improve the Practicality of Software Document Completion Tools"**

This is the supplementary material for the paper "*Properly Offer Options to Improve the Practicality of Software Document Completion Tools*".

It contains:

1. **The complete experimental results for this paper:** 1) the detailed rouge metric scores of using different setups to select and offer options on different models and tasks; 2) the complete experimental result graphs.
2. **The replication package for this paper:** 1) the implementation of  our initial clustering-based option offering method; 2) the scripts to replicate our preliminary evaluation experiments.

------

## Complete Experimental Results

Due to the space limit for the paper, we have not included all the detailed experimental results (the results of T5, and all ROUGE-1 and ROUGE-2 scores of all models) in our paper. Here we release the complete relevant experimental results. Specifically, we provide: (a) the detailed rouge metric scores of using different setups to select and offer options on different models and tasks; (b) the complete experimental result graphs.

### File Structure

```
CompleteResults.zip
│  rouge-metric-scores.csv  # the detailed rouge metric scores of using different setups to select and offer options on different models and tasks.
└─complete-figs  # the complete experimental result graphs for Fig.1 in the paper.
```

## Replication Package

We release the relevant code and data to replicate our experiments. Specifically, we provide: (a) the implementation of  our initial clustering-based option offering method; (b) the instruction and scripts to replicate our preliminary evaluation experiments.


### File Structure

```
ReplicationPackage_Code.zip
└─src
    │  batch-decoding.py
    │  finetune-bart-trainer.py
    │  finetune-t5-trainer.py
    │  ranking.py
    │  select-by-ag-clusting.py
    │  select-by-mmr-ranking.py
    └─evaluate
            rouge-calculator.py
```

### Replication Instruction

0. Install the following necessary Python 3 libraries.

   ```
   datasets==2.3.2
   nltk==3.6.7
   numpy==1.19.5
   pandas==1.1.5
   rouge==1.0.1
   scikit_learn==1.1.2
   torch==1.10.2+cu113
   transformers==4.17.0
   ```

1. Prepare the source datasets. To find the datasets of three different tasks please refer to the original repositories:

   [Stack Overflow Post Title](https://github.com/NTDXYG/SOTitle)

   [Bug Report Title](https://github.com/imcsq/iTAPE)

   [Pull Request Description](https://github.com/Tbabm/PRSummarizer)

   Download the datasets and transfer them to `train.csv`, `valid.csv` and `test.csv` files with columns 'src' and 'target'. As a reminder, we follow the train/valid/test split of the original dataset.

2. Run `src/finetune-t5-trainer.py` or `finetune-bart-trainer.py` to finetune the T5 or BART model.  You can specify the corresponding parameters at the beginning of the scripts, such as the `train_dataset_path` and `valid_dataset_path`.

3. Run `src/batch-decoding.py` to generate the target text outputs with beam search decoding or top-p sampling decoding.

   The target text outputs of the original models please refer to the original repositories mentioned above.

4. Run `src/select-by-ag-clusting.py` to apply our clustering-based option offering method to select the options, and run `select-by-mmr-ranking.py` to select the options using Maximal Marginal Ranking (MMR).

5. Run `src/evaluate/rouge-calculator.py` to calculate the rouge metric scores to measure the performance of our initial clustering-based option offering method and other baselines.
