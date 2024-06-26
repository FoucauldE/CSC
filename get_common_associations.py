import os
import json
import pandas as pd
import argparse
from csc_lib.config import OUTPUT_PATH
from csc_lib.data_loader import load_data
from csc_lib.annotation_processor import get_all_annotations
from csc_lib.association_rules import get_associations

def main(experiment_name, min_docs, min_confidence):

    save_path = os.path.join(OUTPUT_PATH, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    # Load data
    print("Loading data...")
    df_train, df_val, df_gen = load_data()
    df_idx_anns = pd.read_json('DATASET/preprocessed_e3c_cas_2.json')

    # Process annotations
    print("Processing annotations...")
    all_train_anns_flatten, dico_train_anns = get_all_annotations(df_train['fichier'], False)
    all_val_anns_flatten, dico_val_anns = get_all_annotations(df_val['fichier'], False)
    all_gen_anns_flatten, dico_gen_anns = get_all_annotations(df_gen.columns, True)

    # Save processed annotations
    with open(os.path.join(OUTPUT_PATH, 'dico_train_anns_heavy_filter.json'), 'w') as f:
        json.dump(dico_train_anns, f, ensure_ascii=False, indent=4)

    # Get associations
    print("Getting associations...")
    val_rules = get_associations(df_val['fichier'], all_val_anns_flatten, dico_val_anns, min_docs, min_confidence)
    gen_rules = get_associations(df_gen.columns, all_gen_anns_flatten, dico_gen_anns, min_docs, min_confidence)
    train_rules = get_associations(df_train['fichier'], all_train_anns_flatten, dico_train_anns, min_docs, min_confidence)

    # Save associations to csv
    print("Saving results...")
    val_rules.to_csv(os.path.join(save_path, f'FP_growth_val_{min_docs}_docs_{min_confidence}_confidence.csv'))
    gen_rules.to_csv(os.path.join(save_path, f'FP_growth_gen_{min_docs}_docs_{min_confidence}_confidence.csv'))
    train_rules.to_csv(os.path.join(save_path, f'FP_growth_train_{min_docs}_docs_{min_confidence}_confidence.csv'))

    print(f"Task completed successfully. Results are stored in {save_path}/ .")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save the common associations')
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    parser.add_argument('--min_docs', type=int, default=3, help='Minimum number of docs')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold')
    args = parser.parse_args()

    experiment_name, min_docs, min_confidence = args.experiment_name, args.min_docs, args.min_confidence

    print(f"Looking for the most common associations in at least {min_docs} documents with a confidence >= {min_confidence} :")

    main(experiment_name, min_docs, min_confidence)