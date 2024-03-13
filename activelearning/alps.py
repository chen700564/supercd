import random, copy, tqdm, torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.cluster import KMeans
import numpy as np
import torch.nn as nn

def getbatch(features):
    input_ids = []
    attention_masks = []
    for feature in features:
        input_ids.append(feature[0])
        attention_masks.append(feature[1])
    return input_ids,attention_masks

class Sampling:
    def __init__(self,model, modelpath, way):
        self.model = model
        self.modelpath = modelpath
        self.way = way

    def sampling(self, unlabeledindex, datasets, picknum):
        indexs = self.alps(unlabeledindex, datasets, picknum)
        
        result = [datasets[i] for i in indexs]
        return result
    

    def alps(self, unlabeledindex, datasets, picknum):
        '''
        codes are based on https://github.com/forest-snow/alps/blob/03ad15d34cd97f82a4da6aa6fa42c160f4e21637/src/sample.py
        '''
        """Return scores (or vectors) for data [batch] given the active learning method"""

        model = self.model
        tokenizer = self.model.tokenizer
        batchsize = 32

        model.cuda()
        model.eval()
        tokenizerd = []
        selection_pool = []
        for index in range(len(datasets)):
            if index in unlabeledindex:
                instance = datasets[index]
                tokenized_inputs = tokenizer(
                    instance['tokens'], max_length=128, truncation=True, padding='max_length', is_split_into_words=True)
                tokenizerd.append([tokenized_inputs['input_ids'],tokenized_inputs['attention_mask']])
                selection_pool.append({'index':index})
        eval_sampler = SequentialSampler(tokenizerd)
        eval_dataloader = DataLoader(tokenizerd, collate_fn=getbatch, sampler=eval_sampler, batch_size=batchsize)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_dataloader):
                batch = {'input_ids': batch[0],'attention_mask':batch[1]}
                batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

                inputs = {}
                input_ids_cpu = batch['input_ids'].cpu().clone()
                input_ids_mask, labels = self.mask_tokens(input_ids_cpu, tokenizer)

                input_ids = batch['input_ids']
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                inputs["input_ids"] = input_ids
                inputs["labels"] = labels
                inputs["attention_mask"] = batch['attention_mask'].cuda()

                losses = self.get_mlm_loss(model,inputs)
                losses = losses.detach().cpu().numpy().tolist()
                for loss in losses:
                    embeddings.append(loss)
        for i in range(len(embeddings)):
            selection_pool[i]['embedding'] = embeddings[i]
        num_clusters = min(picknum, len(selection_pool))
        return self.k_means_clustering(selection_pool, num_clusters, seed=42)
    
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

    def mask_tokens(self, inputs, tokenizer):
        '''
        codes are based on https://github.com/forest-snow/alps/blob/03ad15d34cd97f82a4da6aa6fa42c160f4e21637/src/sample.py
        '''
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def get_mlm_loss(self, model, inputs, **kwargs):
        '''
        codes are based on https://github.com/forest-snow/alps/blob/03ad15d34cd97f82a4da6aa6fa42c160f4e21637/src/sample.py
        '''
        """Obtain masked language modeling loss from [model] for tokens in [inputs].
        Should return batch_size X seq_length tensor."""
        logits = model(**inputs)[1]
        labels = inputs["labels"]
        batch_size, seq_length, vocab_size = logits.size()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        loss = loss_batched.view(batch_size, seq_length)
        return loss