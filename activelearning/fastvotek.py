import random, copy, os
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch, tqdm
import numpy as np
from collections import defaultdict

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def getsentfeat(dataset,model,tokenizer):
    model.cuda()
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = [tokenizer(' '.join(instance['tokens']), max_length=128) for instance in dataset]

    dataloader = DataLoader(dataset, batch_size=256, collate_fn=data_collator)

    sent_feats = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {k:v.to(model.device) for k,v in batch.items()}
            outputs = model(**batch)
            feats = mean_pooling(outputs.last_hidden_state, batch['attention_mask']).detach().cpu().numpy()
            sent_feats = sent_feats + [i for i in feats]
    sent_feats = np.array(sent_feats)

    return sent_feats

def cos(a,b):
    a2 = a/np.linalg.norm(a,axis=-1,keepdims=True)
    b2 = b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.matmul(a2,b2.T)


class Sampling:
    def __init__(self,model, modelpath, way):
        self.model = model
        self.modelpath = modelpath
        self.way = way

    def sampling(self, unlabeledindex, datasets, picknum):
        indexs = self.fast_votek(self.modelpath, datasets, unlabeledindex ,picknum)
        result = [datasets[i] for i in indexs]
        return result

    def fast_votek(self, modelpath, datasets, unlabelindexs ,picknum, k = 150):
        if not os.path.exists(modelpath + '/sentfeat.npy'):
            embeddings = getsentfeat(datasets,self.model,self.model.tokenizer)
            embeddings = np.array(embeddings)
            os.makedirs(modelpath,exist_ok=True)
            np.save(modelpath+'/sentfeat.npy',embeddings)
        else:
            embeddings = np.load(modelpath+'/sentfeat.npy')
        embeddings = embeddings[unlabelindexs]
        n = len(embeddings)
        bar = tqdm.tqdm(range(n),desc=f'voting')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cos(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
        selected_indices = []
        selected_times = defaultdict(int)
        while len(selected_indices) < picknum:
            cur_scores = defaultdict(int)
            for idx,candidates in votes:
                if idx in selected_indices:
                    cur_scores[idx] = -100
                    continue
                for one_support in candidates:
                    if not one_support in selected_indices:
                        cur_scores[idx] += 10 ** (-selected_times[one_support])
            cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
            selected_indices.append(int(cur_selected_idx))
            for idx_support in vote_stat[cur_selected_idx]:
                selected_times[idx_support] += 1
        print(selected_indices)
        selected_indices = [unlabelindexs[i] for i in selected_indices]
        return selected_indices