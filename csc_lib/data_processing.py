"""
Contains functions used to read a saved dataframe
"""

import pandas as pd
from ast import literal_eval
import re

def preprocess_frozenset_string(s):
    s = re.sub(r'frozenset\({', '{', s)[:-1]
    return s

def convert_to_frozenset(s):
    return frozenset(literal_eval(preprocess_frozenset_string(s)))

def convert_to_literal(s):
    return literal_eval(s)

def correct_literal_eval(df, columns):
    for col in columns:
        if 'frozenset' in str(df[col].iloc[0]):
            df[col] = df[col].apply(convert_to_frozenset)
        else:
            df[col] = df[col].apply(convert_to_literal)
    return df