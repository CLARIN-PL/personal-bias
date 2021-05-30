from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining

import pandas as pd
import numpy as np
import os
import pickle

import torch
device = torch.device("cpu")

from .settings import MAIN_DIR, STORAGE_DIR

def get_embeddings(model, tokenizer, comments, max_seq_len=256):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    all_embeddings = []
    for b_comments in tqdm(batch(comments, 200), total=len(comments)/200):
        
        with torch.no_grad():
            batch_encoding = tokenizer.batch_encode_plus(
                b_comments,
                padding='longest',
                add_special_tokens=True,
                truncation=True, max_length=max_seq_len,
                return_tensors='pt',
            ).to(device)
            emb = model(**batch_encoding)

            
        for i in range(emb[0].size()[0]):
            all_embeddings.append(emb[0][i, batch_encoding['input_ids'][i] > 0, :].mean(axis=0)[None, :])

    return torch.cat(all_embeddings, axis=0)

def prepare_embeddings():

    if os.path.isfile(STORAGE_DIR / 'xlmr_embeddings.p'):
        return

    texts_path = '/mnt/big_one/persemo/user/notebooks/data/splitted_texts.csv'
    texts_multilang_path = '/mnt/big_one/persemo/data/texts/cawi2_texts_multilang.csv'

    texts_df = pd.read_csv(texts_path, sep=',')
    texts_multilang_df = pd.read_csv(texts_multilang_path).fillna('')
    texts_multilang_df = texts_df.merge(texts_multilang_df)

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    model = model.to(device)
    all_embeddings = get_embeddings(model, tokenizer, texts_df.text.tolist(), 256)
    pickle.dump(all_embeddings.cpu().numpy(), open(STORAGE_DIR / 'herbert_embeddings.p', 'wb'))

    tokenizer = AutoTokenizer.from_pretrained("clarin-pl/roberta-polish-kgr10")
    model = AutoModel.from_pretrained("clarin-pl/roberta-polish-kgr10")
    model = model.to(device)
    all_embeddings = get_embeddings(model, tokenizer, texts_df.text.tolist(), 256)
    pickle.dump(all_embeddings.cpu().numpy(), open(STORAGE_DIR / 'polish_roberta_embeddings.p', 'wb'))
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    model = model.to(device)
    all_embeddings = get_embeddings(model, tokenizer, texts_df.text.tolist(), 256)
    pickle.dump(all_embeddings.cpu().numpy(), open(STORAGE_DIR / 'xlmr_embeddings.p', 'wb'))
    for language in [col.split('_')[1] for col in texts_multilang_df.columns[3:]]:
        texts = texts_multilang_df['text_' + language].tolist()
        embeddings = get_embeddings(model, tokenizer, texts, 256)

        pickle.dump(embeddings.cpu().numpy(), open(STORAGE_DIR / f'multilingual/{language}_xlm_embeddings.p', 'wb'))

if __name__ == "__main__":
    prepare_embeddings()