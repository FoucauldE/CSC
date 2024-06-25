import os
import argparse
import pandas as pd
from csc_lib.config import OUTPUT_PATH
from csc_lib.data_loader import load_data
from csc_lib.annotation_processor import get_all_annotations
from csc_lib.tree_builder import  build_tree_recursive, build_tree, get_rare_combinations

def main(max_depth, threshold_nb_docs):

    # Load data
    #
    print("Loading data...")
    df_train, df_val, df_gen = load_data()
    df_idx_anns = pd.read_json('DATASET/preprocessed_e3c_cas_2.json')

    # Process annotations
    print("Processing annotations...")
    all_train_anns_flatten, dico_train_anns = get_all_annotations(df_train['fichier'], False)
    all_val_anns_flatten, dico_val_anns = get_all_annotations(df_val['fichier'], False)
    all_gen_anns_flatten, dico_gen_anns = get_all_annotations(df_gen.columns, True)

    # Format training annotations
    print("Formatting annotations...")
    dico_anns_filtered = {'': []}
    for filename, annotations in dico_train_anns.items():
        for annotation in annotations:
            if annotation not in dico_anns_filtered:
                dico_anns_filtered[annotation] = []
            dico_anns_filtered[annotation].append(filename)
        dico_anns_filtered[''].append(filename)

    # Build a tree representing combinations of annotations
    print("Building combinations tree...")
    root = build_tree(list(dico_anns_filtered.keys()), max_depth=max_depth, threshold_number_docs=threshold_nb_docs, dico_anns_filtered=dico_anns_filtered)

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
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df_rare_combis.to_csv(os.path.join(OUTPUT_PATH, f'rare_combinations_{max_depth}_anns_{threshold_nb_docs}_docs.csv'))

    print(f"Task completed successfully. Results are stored in {OUTPUT_PATH}/.")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Find rare combinations of annotations among training documents.")
    parser.add_argument('-d', '--max_depth', type=int, default=3, help='Maximum depth of constructed tree (ie, max size of combination)')
    parser.add_argument('-t', '--threshold_nb_docs', type=int, default=5, help='Find the combinations present in at most the number of documents specified')
    args = parser.parse_args()

    max_depth, threshold_nb_docs = args.max_depth, args.threshold_nb_docs

    print(f"Looking for rare combinations of size <= {max_depth} present in at most {threshold_nb_docs} training documents :")

    main(max_depth, threshold_nb_docs)


    
