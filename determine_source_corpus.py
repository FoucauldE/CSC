import os
import pandas as pd
import argparse
from csc_lib.data_loader import load_models
from csc_lib.data_processing import correct_literal_eval
from csc_lib.classification import calculate_all_ppls, prepare_data_downsample, train_and_evaluate_model
from csc_lib.config import FT_MODEL_PATH, OUTPUT_PATH, models_and_params

def main(train_rules_path, unseen_rules_path, experiment_name, base_model_path, ft_model_path=FT_MODEL_PATH):

    # Check if association rules paths exist
    if not os.path.isfile(train_rules_path):
        raise FileNotFoundError(f"The specified path for CSV file containing association rules is not correct: {train_rules_path}")
    if not os.path.isfile(unseen_rules_path):
        raise FileNotFoundError(f"The specified path for CSV file containing association rules is not correct: {unseen_rules_path}")
    
    # Creates save path
    save_path = os.path.join(OUTPUT_PATH, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Load association rules
    train_rules = pd.read_csv(train_rules_path, index_col=0)
    train_rules = correct_literal_eval(train_rules, ['antecedents', 'consequents'])
    unseen_rules = pd.read_csv(unseen_rules_path, index_col=0)
    unseen_rules = correct_literal_eval(unseen_rules, ['antecedents', 'consequents'])

    # Prepare base model and fine-tuned model to measure their perplexity
    tokenizer, base_model, ft_model = load_models(ft_model_path, base_model_path)

    # Calculate perplexities for each model
    # Currently we only consider associations of 3 annotations
    ppls_train_data = calculate_all_ppls(train_rules, base_model, ft_model, tokenizer, len_associations=3)
    ppls_unseen_data = calculate_all_ppls(unseen_rules, base_model, ft_model, tokenizer, len_associations=3)

    # Split into training and testing set after balancing classes by downsampling
    X_train, X_test, y_train, y_test = prepare_data_downsample(ppls_unseen_data, ppls_train_data)

    # Train ML models specified in config.py
    classification_results = {}
    for model_name, model_info in models_and_params.items():
        print(f"Training {model_name}...", end=' ')
        result = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            model_info['model'], model_info['param_grid'],
            os.path.join(OUTPUT_PATH, experiment_name, f'ROC_curve_{model_name}.png')
        )
        print("✔")
        classification_results[model_name] = result


    # Save the results
    with open(f"{save_path}/classification_scores.txt", "a") as f:
        for model, metrics in classification_results.items():
            conf_matrix, acc, roc_auc, precision, tpr_at_fpr = metrics.values()
            f.write(f"Model: {model}\n")
            f.write(f"Confusion matrix:\n{conf_matrix}\n")
            f.write(f"Accuracy: {acc:.3f}\n")
            f.write(f"roc auc: {roc_auc:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"TPR at FPR: {', '.join(f'{k}: {v:.3f}' for k, v in tpr_at_fpr.items())}\n")
            f.write("\n\n")

    print(f"Task completed successfully. Results are stored in {save_path}/classification_scores.txt .")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Measure to which extent a model can generate consequents when prompting antecedents')
    parser.add_argument('-t', '--train_rules_path', type=str, required=True, help='Path to the association rules csv file of training data')
    parser.add_argument('-u', '--unseen_rules_path', type=str, required=True, help='Path to the association rules csv file of unseen data')
    parser.add_argument('-b', '--base_model', type=str, default='bigscience/bloom-1b1', help='Base model path')
    parser.add_argument('-ft', '--ft_model_path', type=str, default=FT_MODEL_PATH, help='Path to the fine-tuned model')
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    args = parser.parse_args()

    train_rules_path, unseen_rules_path, experiment_name, base_model_path, ft_model_path = args.train_rules_path, args.unseen_rules_path, args.experiment_name, args.base_model, args.ft_model_path

    main(train_rules_path, unseen_rules_path, experiment_name, base_model_path, ft_model_path=FT_MODEL_PATH)