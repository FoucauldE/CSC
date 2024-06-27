import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def get_associations(set_all_anns:set, anns_dict:dict, min_docs:int, min_confidence:float):
    """
    Input:
    set_all_anns (set): A set of all unique filtered annotations in the specified folder of .ann files.
    anns_dict (dict): A dictionary where keys are filenames and values are lists of annotations.
    min_docs (int): The minimum number of documents an annotation must appear in to be considered.
    min_confidence (float): The minimum confidence required for an association rule.

    Output:
    rules (pd.DataFrame): A DataFrame of association rules.
    """

    # Get a list of all filenames from the dictionary
    filenames = list(anns_dict.keys())

    # Encode the annotations as a list of dictionaries, where each dictionary represents a file
    # and contains annotations as keys and boolean values indicating presence/absence as values
    encoded_anns = [{ann: (ann in anns_dict[filename]) for ann in set_all_anns} for filename in filenames]
    one_hot_df = pd.DataFrame(encoded_anns)

    # Calculate the minimum support required for an annotation to be considered frequent
    min_support = min_docs / len(filenames)

    # Use the fpgrowth algorithm to find frequent itemsets (annotations)
    frequent_itemsets = fpgrowth(one_hot_df, min_support=min_support, use_colnames=True)

    # Generate association rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, min_threshold=min_confidence)

    return rules