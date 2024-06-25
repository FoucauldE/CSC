import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def get_associations(filenames:list, set_all_anns, dico_anns, min_docs, min_confidence):

    encoded_anns = [{ann: (ann in dico_anns[filename]) for ann in set_all_anns} for filename in filenames]
    one_hot_df = pd.DataFrame(encoded_anns)
    min_support = min_docs / len(filenames)
    frequent_itemsets = fpgrowth(one_hot_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, min_threshold=min_confidence)
    return rules