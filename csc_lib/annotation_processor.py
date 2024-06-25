import os
from tqdm import tqdm
import pandas as pd
from csc_lib.config import ANN_PATH, TYPES_TO_KEEP, FILTER_OUT

def get_all_annotations(filenames, is_generated):
    """
    Returns a set with PROC, DISO & CHEM annotations over all documents.
    At the scale of a document, when the spans of 2 annotations overlap, only the largest annotation is kept.
    """
    
    ann_path = ANN_PATH
    kept_annotations = []
    dico_anns = {filename: [] for filename in filenames}
    df_idx_anns = pd.read_json('DATASET/preprocessed_e3c_cas_2.json')

    for filename in tqdm(filenames, desc="Processing annotations"):
        # Study the right portion of the .ann file, in case different files are described in a single .ann file
        idx_start, idx_end = float('-inf'), float('inf')
        annotations_filename = filename
        # print(annotations_filename)

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
        