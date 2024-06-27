# Confidentiality of Synthetic Corpora

While many domains such as e-commerce, social media, and finance have leveraged vast amounts of available data to make significant advancements, sensitive domains like health and banking face substantial challenges due to the lack of shareable corpora. The CoDeinE project aims to address this issue by generating synthetic texts that replicate the linguistic properties of real documents while preserving confidentiality. Given the risks of re-identification from minimal information, we propose a method to evaluate the confidentiality of such synthetic data within a medical context.

## Requirements

To run the code you can create a new environment and install the required libraries by running the following commands after cloning the repository:

```bash
    python -m venv csc_env
    source csc_env/bin/activate
    pip install -r requirements.txt
```

## How to run the code

### 0. Load the data

1. Insert your data in the `DATASET/` folder.
2. Configure the `csc_lib/config.py` as needed to match your requirements.

### 1. Evaluate the confidentality of the corpora

#### Training data

To ensure that no sensitive combination of associations relates to a low number of training documents, you can run the following command:

   ```bash
   python get_rare_combinations.py -p path/to/anns/folder -e experiment_1 --max_depth 3 --threshold_nb_docs 4
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
   python get_common_associations.py -p path/to/anns/folder -e experiment_2 --min_docs 3 --min_confidence 0.7
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
python consequents_generation.py -e Test_general1 -p path_association_rules.csv -n 30 -t 200 -s 1000 -b 2 -m path/to/model
```

With:
- `-p`, `--path_rules`: Path to the association rules CSV file to study.
- `-e`, `--experiment_name`: Name of the current experiment.
- `-n`, `--number_tries`: Number of generations for each studied case.
- `-t`, `--max_tokens`: Number of new generated tokens.
- `-s`, `--step`: Step size between blocks.
- `-b`, `--block_size`: Block size.
- `-m`, `--ft_model_path`: Path to the fine-tuned model.


The outputs are :
- a .csv file, detailing the number of consequents found for each try and the list of generated consequents during the most successful try
- a .png file, showcasing the distribution of the number of consequents found for each try over each studied association. The color of the violin plots indicates the percentage of consequents found during the most successful try.