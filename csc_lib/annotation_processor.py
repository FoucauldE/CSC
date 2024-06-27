import os
from tqdm import tqdm
import pandas as pd
from csc_lib.config import TYPES_TO_KEEP, FILTER_OUT

def load_annotations_from_folder(folder_path):
    """
    Returns a dictionary where each key is a filename and each value is the list of annotations found in that file.
    Annotations are filtered depending on their type (PROC, DISO...) and their span.
    At the scale of a document, when the spans of 2 annotations overlap, only the largest annotation is kept.
    """
    
    # kept_annotations = []
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.ann')]
    dico_anns = {filename: [] for filename in filenames}

    for filename in tqdm(filenames, desc="Processing annotations..."):

        with open(os.path.join(folder_path, f"{filename}"), 'r') as f:

            # Store all annotations
            annotations = []
            for line in f:
                line = line.strip().split('\t')
                type_ann, start_ann, end_ann = line[1].split()
                annotation = line[-1]
                if type_ann in TYPES_TO_KEEP:
                    annotations.append((int(start_ann), int(end_ann), annotation))
                
            # Filter annotations depending on their type (PROC, DISO...) and their span
            # In case of overlapping spans (ie, they relate to the same portion of text), we only keep the largest one
            if annotations:
                current_start, current_end, current_annotation = annotations[0]
                for i, (start, end, annotation) in enumerate(annotations[1:]):
                    if start >= current_end:
                        if current_annotation not in FILTER_OUT:
                            dico_anns[filename].append(current_annotation)
                        current_start, current_end, current_annotation = start, end, annotation
                    elif len(annotation) > len(current_annotation):
                        current_start, current_end, current_annotation = start, end, annotation

                if current_annotation not in FILTER_OUT:
                    dico_anns[filename].append(current_annotation)

    return dico_anns


def flatten_annotations_dict(dict_anns):
    """
    Flattens a dictionary of annotations to a set of unique annotations.

    Input: A dictionary where each key is a filename and each value is a list of annotations found in that file.
    Output: A set of unique annotations.
    """
    return set(annotation for annotations in dict_anns.values() for annotation in annotations)


def reverse_annotations_dict(dico_anns:dict):
    """
    Input: A dictionary where each key is a filename and each value is the (filtered) list of annotations found in that file.
    Output: A dictionary where each key is an annotation and each value is a list of filenames where that annotation was found.
    """
    reversed_dico_anns = {'': []}
    for filename, annotations in tqdm(dico_anns.items(), desc='Formatting annotations...'):
        for annotation in annotations:
            if annotation not in reversed_dico_anns:
                reversed_dico_anns[annotation] = []
            reversed_dico_anns[annotation].append(filename)
        reversed_dico_anns[''].append(filename)
    return reversed_dico_anns


def build_df_with_indications(df:pd.DataFrame):
    """
    Returns a dataframe containing filename, age, sex, annotation and indication
    Indication is a description of the main reason of the patient's arrival
    """

    def get_indications(filename):
        indication = None
        with open(os.path.join(ANN_PATH, f"{filename}.ann"), 'r') as f:
            for line in f:
                if 'origine' in line:
                    indication = line.strip().split('\t')[-1]
        return indication

    columns_to_keep = ['fichier', 'age', 'sexe', 'annotations']
    new_rows = []

    for _, row in df.iterrows():
        filename = row['fichier']
        if filename[:4] == 'file':
            indication = get_indications(filename)
            if indication is not None:
                new_rows.append(list(row[columns_to_keep]) + [indication])

    return pd.DataFrame(new_rows, columns = columns_to_keep + ['indication'])



"""
def old_get_all_annotations(filenames, is_generated):
    # OLD VERSION (less general)
    # Returns a set with PROC, DISO & CHEM annotations over all documents.
    # At the scale of a document, when the spans of 2 annotations overlap, only the largest annotation is kept.
    
    
    ann_path = ANN_PATH
    kept_annotations = []
    dico_anns = {filename: [] for filename in filenames}
    df_idx_anns = pd.read_json('DATASET/preprocessed_e3c_cas_2.json')

    for filename in tqdm(filenames, desc="Processing annotations"):
        # Study the right portion of the .ann file, in case different files are described in a single .ann file
        idx_start, idx_end = float('-inf'), float('inf')
        annotations_filename = filename

        if not is_generated:
            if filename[:4] == 'file':
                ann_path = "DATASET/corpus_annotes/CAS"
            else:
                ann_path = "DATASET/corpus_annotes/E3C"

            multi_infos = df_idx_anns[filename]['multi_infos']
            
            if not pd.isna(multi_infos):
                annotations_filename, span = multi_infos['original_file'], multi_infos['span']
                idx_start, idx_end = map(int, span)


        with open(os.path.join(ann_path, f"{annotations_filename}.ann"), 'r') as f:

            annotations = []
            for line in f:
                line = line.strip().split('\t')
                type_ann, start_ann, end_ann = line[1].split()
                annotation = line[-1]

                if idx_start <= int(start_ann) and int(end_ann) <= idx_end:
                    if type_ann in TYPES_TO_KEEP:
                        annotations.append((int(start_ann), int(end_ann), annotation))
                
            current_start, current_end, current_annotation = annotations[0]
            for i, (start, end, annotation) in enumerate(annotations[1:]):
                if start >= current_end:
                    if current_annotation not in FILTER_OUT:
                        dico_anns[filename].append(current_annotation)
                    kept_annotations.append((current_start, current_end, current_annotation))
                    current_start, current_end, current_annotation = start, end, annotation
                elif len(annotation) > len(current_annotation):
                    current_start, current_end, current_annotation = start, end, annotation

            if current_annotation not in FILTER_OUT:
                dico_anns[filename].append(current_annotation)
            kept_annotations.append((current_start, current_end, current_annotation))

    all_flatten_anns_to_keep = set([a[2] for a in kept_annotations if a[2] not in FILTER_OUT])

    return all_flatten_anns_to_keep, dico_anns
"""