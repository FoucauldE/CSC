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
3. It is assumed that you work with a training, a validation and a generated dataset. If not, also modify the `csc_lib/data_loader.py` and the dependent functions accordingly.

### 1. Evaluate the confidentality of the corpora

#### Training data

To ensure that no sensitive combination of associations relates to a low number of training documents, you can run the following command:

   ```bash
   python get_rare_combinations.py -e experiment_1 --max_depth 3 --threshold_nb_docs 4
   ```
Where :
- `-e` specifies the name of the current experiment
- `--max_depth 3` or `-d 3` specifies that we analyze combinations of **up to 3 annotations**
- `--threshold_nb_docs 4` or `-t 4` specifies that we want to focus on combinations that appear in **at most 4 training documents**.


#### Synthetic data

To ensure that strong associations of annotations in the synthetic corpus do not compromise sensitive information, you can run the following command:

   ```bash
   python get_common_associations.py -e experiment_2 --min_docs 3 --min_confidence 0.7
   ```
Where :
- `-e` specifies the name of the current experiment
- `--min_docs 3` specifies that we analyze associations that appear in **at least 3** synthetic documents
- `--min_confidence 0.7` specifies that we only consider associations with a **confidence level of 0.7 or higher**, indicating a strong correlation between the annotations.

### 2. 