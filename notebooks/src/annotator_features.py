import numpy as np
import pandas as pd
from scipy.stats import entropy
import simpledorff
from simpledorff.metrics import interval_metric

# def get_entropy(labels, base=None, contr_type='std'):
#     value, counts = np.unique(labels, return_counts=True)

#     if contr_type == 'std':
#         en = entropy(counts, base=base)
    
#     return en


def get_controversy(labels, contr_type='std'):
    if contr_type == 'std':
        en = np.std(labels)
    # else:
    #     en = 
    
    return en

def get_text_entropies(annotations_df, columns, contr_type):
    text_entropies = annotations_df.groupby('text_id').apply(lambda g: pd.Series([get_controversy(g[col].values, contr_type=contr_type) 
                                                                                  for col in columns]))
    
    text_entropies.columns = [col + '_entropy' for col in columns]
    
    text_entropies['mean_entropy'] = text_entropies.values.mean(axis=1)
    text_entropies['max_entropy'] = text_entropies.values.max(axis=1)
    return text_entropies

def get_random_annotations(annotations_df, number=None):
    df = annotations_df.groupby('annotator_id').apply(lambda g: g.sample(frac=1).iloc[:number]).reset_index(drop=True)

    return df

def get_most_controversial_annotations(annotations_df, columns, number=None, contr_type='std'):
    text_entropies = get_text_entropies(annotations_df, columns, contr_type=contr_type)
    df = annotations_df.merge(text_entropies, on='text_id')
    
    if number is not None:
        df = df.groupby('annotator_id').apply(lambda g: g.sort_values(by='mean_entropy', ascending=False).iloc[:number]).reset_index(drop=True)
    else:
        df = df.groupby('annotator_id').apply(lambda g: g.sort_values(by='mean_entropy', ascending=False)).reset_index(drop=True)
    
    return df


def get_annotator_biases(annotations_df, columns):
    text_means = annotations_df.groupby('text_id').mean().loc[:, columns]
    text_stds = annotations_df.groupby('text_id').std().loc[:, columns]
    
    annotations_df = annotations_df.join(text_means, rsuffix='_mean', on='text_id').join(text_stds, rsuffix='_std', on='text_id')
    
    for col in columns:
        annotations_df[col + '_z_score'] = (annotations_df[col] - annotations_df[col + '_mean']) / annotations_df[col + '_std']
    
    annotator_biases = annotations_df.groupby('annotator_id').mean().loc[:, [col + '_z_score' for col in columns]]

    annotator_biases.columns = [col + '_bias' for col in columns]
    
    return annotator_biases