# Confidentiality of Synthetic Corpora

While many domains such as e-commerce, social media and finance have leveraged vast amounts of available data to make significant advancements, sensitive domains like health and banking face substantial challenges due to the lack of shareable corpora. The CoDeinE project aims to address this issue by generating synthetic clinical texts that replicate the linguistic properties of real documents while preserving confidentiality. Given the risks of re-identification from minimal information, we propose a method to evaluate the confidentiality of such synthetic data within a medical context.

## Requirements

To run the code you can create a new environment and install the required libraries by running the following commands after cloning the repository:

```bash
    python -m venv csc_env
    source csc_env/bin/activate
    pip install -r requirements.txt
```

## How to run the code

### 0. Load the data

1. Insert your data in the `DATASET/` folder by creating subfolders corresponding to the training, validation and synthetic corpus, respectively. Annotations are expected to be stored in .ann files using the BRAT format.
2. Configure the `csc_lib/config.py` as needed to match your requirements (default paths, annotations to filter, generation parameters...)

### 1. Evaluate the confidentality of the corpora

#### Training data

To ensure that no sensitive combination of associations relates to a low number of training documents, you can run the following command:

   ```bash
   python get_rare_combinations.py -e experiment_1 -p path/to/anns/folder  --max_depth 3 --threshold_nb_docs 4
   ```
Where :
- `-p` specifies the path of the folder containing the .ann files to study
- `-e` specifies the name of the current experiment
- `--max_depth 3` or `-d 3` specifies that we analyze combinations of **up to 3 annotations**
- `--threshold_nb_docs 4` or `-t 4` specifies that we want to focus on combinations that appear in **at most 4 training documents**.

Results are stored in a .csv file containing the rare combinations, along with the number of documents and annotations involved, ready to be analyzed.

#### Synthetic data

To ensure that strong associations of annotations in the synthetic corpus do not compromise sensitive information, you can run the following command:

   ```bash
   python get_common_associations.py -e experiment_2 -p path/to/anns/folder  --min_docs 3 --min_confidence 0.7
   ```
Where :
- `-p` specifies the path of the folder containing the .ann files to study
- `-e` specifies the name of the current experiment
- `--min_docs 3` specifies that we analyze associations that appear in **at least 3** synthetic documents
- `--min_confidence 0.7` specifies that we only consider associations with a **confidence level of 0.7 or higher**, indicating a strong correlation between the annotations.

The output is a .csv file containing the association rules (antecedents-consequents), along with various metrics such as support, confidence, and lift.

### 2. Measure the extent to which a model can generate consequents associated with prompted antecedents

To study if a model could generate consequents associated with prompted antecedents, you can follow these steps:

1. **Identify common associations**: Run `get_common_associations.py` described in the previous part
2. **Evaluate consequents generation**: Run the following command:

```bash
python consequents_generation.py -e experiment_3 -p path_association_rules.csv -n 30 -t 200 -s 1000 -b 2 -ft path/to/ft/model
```

With:
- `-p`, `--path_rules`: Path to the CSV file containing association rules to study.
- `-e`, `--experiment_name`: Name of the current experiment.
- `-n`, `--number_tries`: Number of attempts to generate consequents for each association.
- `-t`, `--max_tokens`: Number of new generated tokens.
- `-s`, `--step`: Spacing between the starting points of consecutive sub-blocks of association rules. For example, with `-s 1000`, the process would start at rule 0, skip 999 rules, and start the next sub-block at rule 1000.
- `-b`, `--block_size`: Number of association rules that compose a single block.
- `-ft`, `--ft_model_path`: Path to the fine-tuned model.


The outputs are :
- a .csv file, detailing the number of consequents found for each try and the list of generated consequents during the most successful try
- a .png file, showcasing the distribution of the number of consequents found for each try over each studied association. The color of the violin plots indicates the percentage of consequents found during the most successful try.

### 3. Membership Inference Attack

To study if an attacker could determine if a piece of data was used during the training of the language model used to generate synthetic data, we propose to compare the perplexities obtained with a base model and a model fine-tuned on the synthetic data. From these perplexities, could a Machine Learning model determine the source corpus of a given piece of data ? To study this question, you can follow these steps:

1. Identify associations from the training corpus by running `get_common_associations.py`.
2. Repeat the experiment on another corpus, not used for the training of our generative model.
3. Determine if ML models can determine the source corpus of associations from the corresponding perplexities obtained with 2 models by running the following command:

```bash
python determine_source_corpus.py -e experiment_name -t Outputs/ppls_train.csv -u Outputs/ppls_unseen.csv  -ft path/to/ft_model -b path/to/base_model
```

With:
- `-t`, `--train_rules_path`: Path to the association rules CSV file from the training corpus.
- `-u`, `--unseen_rules_path`: Path to the association rules CSV file from a corpus not used for training.
- `-e`, `--experiment_name`: Name of the current experiment.
- `-ft`, `--ft_model_path`: Path to the fine-tuned model.
- `-b`, `--base_model`: Path to the base model.

The outputs are :
- a .txt file detailing the classification metrics obtained when training the models specified in `config.py`
- .png files with the ROC Curves obtained with each model used.
