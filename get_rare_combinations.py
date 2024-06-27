import os
import argparse
import pandas as pd
from csc_lib.config import OUTPUT_PATH
from csc_lib.annotation_processor import load_annotations_from_folder, reverse_annotations_dict
from csc_lib.tree_builder import  build_tree_recursive, build_tree, get_rare_combinations

def main(ann_path, experiment_name, max_depth, threshold_nb_docs):

    # Process annotations
    dict_annotations = load_annotations_from_folder(ann_path)

    # Format training annotations
    dict_annotations_reversed = reverse_annotations_dict(dict_annotations)

    # Build a tree representing combinations of annotations
    print("Building combinations tree... (This step can be long)")
    root = build_tree(list(dict_annotations_reversed.keys()), max_depth=max_depth, threshold_number_docs=threshold_nb_docs, dico_anns_filtered=dict_annotations_reversed)

    # Format and save the output
    print("Identifying combinations...")
    rare_combinations = get_rare_combinations(
        tree_dict = root.to_dict(),
        threshold_nb_docs = threshold_nb_docs,
        max_combination_size = max_depth
    )

    df_rare_combis = pd.DataFrame(rare_combinations)
    df_rare_combis['# docs'] = df_rare_combis.apply(lambda x: len(x['docs']), axis=1)
    df_rare_combis['# annotations'] = df_rare_combis.apply(lambda x: len(x['combination']), axis=1)
    df_rare_combis.sort_values(by=['# docs', '# annotations'], ascending=False, inplace=True)

    print("Saving results...")
    save_path = os.path.join(OUTPUT_PATH, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    df_rare_combis.to_csv(os.path.join(save_path, f'rare_combinations_{max_depth}_anns_{threshold_nb_docs}_docs.csv'))

    print(f"Task completed successfully. Results are stored in {save_path}/.")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Find rare combinations of annotations among training documents.")
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the folder containing .ann files')
    parser.add_argument('-e', '--experiment_name', type=str, required=True)
    parser.add_argument('-d', '--max_depth', type=int, default=3, help='Maximum depth of constructed tree (ie, max size of combination)')
    parser.add_argument('-t', '--threshold_nb_docs', type=int, default=5, help='Find the combinations present in at most the number of documents specified')
    args = parser.parse_args()

    path, experiment_name, max_depth, threshold_nb_docs = args.path, args.experiment_name, args.max_depth, args.threshold_nb_docs

    print(f"Looking for rare combinations of size <= {max_depth} present in at most {threshold_nb_docs} training documents :")

    main(path, experiment_name, max_depth, threshold_nb_docs)


    
