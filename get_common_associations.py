import os
import argparse
from csc_lib.config import OUTPUT_PATH
from csc_lib.annotation_processor import load_annotations_from_folder, flatten_annotations_dict
from csc_lib.association_rules import get_associations

def main(ann_path, experiment_name, min_docs, min_confidence):

    # Process annotations
    dict_annotations = load_annotations_from_folder(ann_path)
    all_annotations_flatten = flatten_annotations_dict(dict_annotations)

    # Get associations rules
    print("Getting associations rules...")
    association_rules = get_associations(all_annotations_flatten, dict_annotations, min_docs, min_confidence)

    # Save associations to csv
    print("Saving results...")
    save_path = os.path.join(OUTPUT_PATH, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    association_rules.to_csv(os.path.join(save_path, f'FP_growth_{min_docs}_docs_{min_confidence}_confidence.csv'))

    print(f"Task completed successfully. Results are stored in {save_path}/ .")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save the common associations')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the folder containing .ann files')
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    parser.add_argument('--min_docs', type=int, default=3, help='Minimum number of docs')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Minimum confidence threshold')
    args = parser.parse_args()

    path, experiment_name, min_docs, min_confidence = args.path, args.experiment_name, args.min_docs, args.min_confidence

    print(f"Looking for the most common associations in at least {min_docs} documents with a confidence >= {min_confidence} :")

    main(path, experiment_name, min_docs, min_confidence)