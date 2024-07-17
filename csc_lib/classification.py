import torch
import math
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, make_scorer, roc_curve
import matplotlib.pyplot as plt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_ppl(inputs, selected_model, tokenizer):
    """
    Returns the perplexity of the selected model for the given inputs
    """

    inputs = tokenizer(inputs, return_tensors='pt').to('cuda')
    loss = selected_model(
        input_ids = inputs['input_ids'],
        labels = inputs['input_ids']
    ).loss
    ppl = torch.exp(loss)
    return ppl.item()


def get_associations(df:pd.DataFrame, len_associations=3):
    """
    Yields associations (antecedents + consequents) from a dataframe in order to iterate over them.
    By default, we consider associations of 3 annotations
    """

    for _, row in df.iterrows():
        antecedents, consequents = list(row['antecedents']), list(row['consequents'])
        if len(antecedents) + len(consequents) == len_associations:
            yield antecedents + consequents


def count_considered_associations(df: pd.DataFrame, len_associations=3):
    """
    Counts the number of associations that meet the length requirement.
    Used for display purposes when using tqdm bar.
    """
    count = 0
    for _, row in df.iterrows():
        antecedents, consequents = list(row['antecedents']), list(row['consequents'])
        if len(antecedents) + len(consequents) == len_associations:
            count += 1
    return count


def calculate_all_ppls(df:pd.DataFrame, base_model, ft_model, tokenizer, len_associations=3):
    """
    Returns a dataframe containing, for each association, the perplexity obtained with 2 distinct models and the ratio of their log-perplexity
    """

    # Counts the associations that meet the length requirement (default: 3)
    considered_associations_count = count_considered_associations(df, len_associations)

    result_df = pd.DataFrame(columns=['association', 'base_ppl', 'finetuned_ppl', 'ratio_ppls'])
    for input in tqdm(get_associations(df, len_associations), desc=f'Calculating perplexities...', total=considered_associations_count):
        input = ', '.join(input)
        ppls = [get_ppl(input, selected_model, tokenizer) for selected_model in [base_model, ft_model]]
        result_df.loc[len(result_df)] = [input, ppls[0], ppls[1], math.log(ppls[0])/math.log(ppls[1])]
    return result_df


def prepare_data_downsample(df0, df1):
    """
    Train test split on balanced data (balanced by downsampling the larger df)
    """

    df0['label'] = 0
    df1['label'] = 1

    if len(df0) < len(df1):
        smaller_df = df0
        larger_df = df1
    else:
        smaller_df = df1
        larger_df = df0

    downsampled_df = resample(larger_df,
                              replace=False,
                              n_samples=len(smaller_df),
                              random_state=42)    

    df_classification = pd.concat([smaller_df, downsampled_df])

    # X = df_classification[['base_ppl', 'finetuned_ppl', 'ratio_ppls']]
    X = df_classification[['ratio_ppls']]
    y = df_classification[['label']]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.flatten(), test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grid, save_fig_path, use_smote=False):
    """
    Train specified model and return results
    Apply grid search on precision
    """

    # Currently not used (to augment the data)
    if use_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])

    """
    # Precision is set to 0 when no positive prediction is made
    def custom_precision_score(y_true, y_pred):
        return precision_score(y_true, y_pred, zero_division=0)
    custom_precision = make_scorer(custom_precision_score)
    """

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Define FPR threshold to consider
    fpr_thresholds = [0.01, 0.05, 0.1]
    tpr_at_fpr = {}

    for threshold in fpr_thresholds:
        # Find the TPR corresponding FPR threshold studied
        # Note that we do not use zip to directly build the dictionary since the same FPR could appear more than once
        idx = next(i for i, f in enumerate(fpr) if f > threshold) - 1
        tpr_at_fpr[threshold] = tpr[idx] if idx > 0 else 0
        
    # Plot and save the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve using {model.__class__.__name__}')
    plt.grid(True)
    plt.savefig(save_fig_path)

    return {
        'conf_matrix': conf_matrix,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'tpr_at_fpr': tpr_at_fpr
    }
