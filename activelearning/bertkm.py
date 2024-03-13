import random, copy, os, tqdm
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

class Sampling:
    def __init__(self,model, modelpath, way):
        self.model = model
        self.modelpath = modelpath
        self.way = way

    def sampling(self, unlabeledindex, datasets, picknum,):
        indexs = self.bertkm(datasets,picknum,unlabeledindex)
        result = [datasets[i] for i in indexs]
        return result
    
    def bertkm(self,datasets,picknum,unlabeledindex):
        '''
        Implementation of BERT-KM introduced in:
            https://aclanthology.org/2020.emnlp-main.637.pdf
            
        codes are baed on https://github.com/nlp-uoregon/famie/blob/main/src/famie/api/active_learning/utils.py
        '''
        model = self.model.cuda()
        model.eval()

        instances = []
        for instance in datasets:
            instances.append(model.tokenizer(' '.join(instance['tokens']), max_length=128))
        
        batchsize = 32
    
        dataloader = DataLoader(instances, batch_size=batchsize, collate_fn=DataCollatorWithPadding(model.tokenizer))

        selection_pool = []
        sent_feats = []


        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                feats = outputs.last_hidden_state[:,0].detach().cpu().numpy()
                sent_feats = sent_feats + [i for i in feats]

        for index in range(len(sent_feats)):
            if index in unlabeledindex:
                selection_pool.append({'index':index,'embedding':sent_feats[index]})

        num_clusters = min(picknum, len(selection_pool))
        return self.k_means_clustering(selection_pool, num_clusters, seed=2333)

    def k_means_clustering(self, selection_pool, num_clusters, init_centroids='random', seed=None):
        '''
        Modified from https://github.com/JordanAsh/badge/blob/master/query_strategies/kmeans_sampling.py and https://github.com/nlp-uoregon/famie/blob/main/src/famie/api/active_learning/utils.py
        '''
        embeds_unlabeled = np.array([ex['embedding'] for ex in selection_pool])
        cluster_learner = KMeans(n_clusters=num_clusters, init=init_centroids, random_state=seed)
        print('kmeans')
        cluster_learner.fit(embeds_unlabeled)
        print('finish fitting')

        cluster_idxs = cluster_learner.predict(embeds_unlabeled)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeds_unlabeled - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = set(np.array(
            [np.arange(embeds_unlabeled.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in
            range(num_clusters)]).tolist())
        return [selection_pool[idx]['index'] for idx in q_idxs]