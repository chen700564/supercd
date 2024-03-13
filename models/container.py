

'''
based on https://github.com/psunlpgroup/CONTaiNER/blob/main/src/utils.py and https://github.com/psunlpgroup/CONTaiNER/blob/main/src/container.py
'''

import os
import logging
import json
from transformers import Trainer,BertPreTrainedModel,BertModel,AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch,tqdm
import gc
import torch.nn.functional as F
logging.basicConfig(level = logging.INFO)


START_ID = 0
O_ID = 1

class ViterbiDecoder:
    """
    Generalized Viterbi decoding
    https://github.com/asappresearch/structshot/blob/2bf53794b3ffd55b9970eb7e3c4b68847b1bd4eb/structshot/viterbi.py
    """

    def __init__(self, n_tag, abstract_transitions, tau):
        """
        We assume the batch size is 1, so no need to worry about PAD for now
        n_tag: START, O, and I_Xs
        """
        super().__init__()
        self.transitions = self.project_target_transitions(n_tag, abstract_transitions, tau)

    @staticmethod
    def project_target_transitions(n_tag, abstract_transitions, tau):
        s_o, s_i, o_o, o_i, i_o, i_i, x_y = abstract_transitions
        # self transitions for I-X tags
        a = torch.eye(n_tag) * i_i
        # transitions from I-X to I-Y
        b = torch.ones(n_tag, n_tag) * x_y / (n_tag - 3)
        c = torch.eye(n_tag) * x_y / (n_tag - 3)
        transitions = a + b - c
        # transition from START to O
        transitions[START_ID, O_ID] = s_o
        # transitions from START to I-X
        transitions[START_ID, O_ID+1:] = s_i / (n_tag - 2)
        # transition from O to O
        transitions[O_ID, O_ID] = o_o
        # transitions from O to I-X
        transitions[O_ID, O_ID+1:] = o_i / (n_tag - 2)
        # transitions from I-X to O
        transitions[O_ID+1:, O_ID] = i_o
        # no transitions to START
        transitions[:, START_ID] = 0.

        powered = torch.pow(transitions, tau)
        summed = powered.sum(dim=1)

        transitions = powered / summed.view(n_tag, 1)

        transitions = torch.where(transitions > 0, transitions, torch.tensor(.000001))

        #print(transitions)
        #print(torch.sum(transitions, dim=1))
        return torch.log(transitions)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Take the emission scores calculated by NERModel, and return a tensor of CRF features,
        which is the sum of transition scores and emission scores.
        :param scores: emission scores calculated by NERModel.
            shape: (batch_size, sentence_length, ntags)
        :return: a tensor containing the CRF features whose shape is
            (batch_size, sentence_len, ntags, ntags). F[b, t, i, j] represents
            emission[t, j] + transition[i, j] for the b'th sentence in this batch.
        """
        batch_size, sentence_len, _ = scores.size()
        # print(scores)
        # expand the transition matrix batch-wise as well as sentence-wise
        transitions = self.transitions.expand(batch_size, sentence_len, -1, -1)

        # add another dimension for the "from" state, then expand to match
        # the dimensions of the expanded transition matrix above
        emissions = scores.unsqueeze(2).expand_as(transitions)

        # add them up

        # print(transitions)
        # print(emissions)

        return transitions + emissions

    @staticmethod
    def viterbi(features: torch.Tensor) -> torch.Tensor:
        """
        Decode the most probable sequence of tags.
        Note that the delta values are calculated in the log space.
        :param features: the feature matrix from the forward method of CRF.
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: a tensor containing the most probable sequences for the batch.
            shaped (batch_size, sentence_len)
        """
        batch_size, sentence_len, ntags, _ = features.size()

        # initialize the deltas
        delta_t = features[:, 0, START_ID, :]
        deltas = [delta_t]

        # use dynamic programming to iteratively calculate the delta values
        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)

        # now iterate backward to figure out the most probable tags
        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)

def get_abstract_transitions(instances):
    """
    Compute abstract transitions on the training dataset for StructShot
    """

    entityname = 'entity'
    if 'entity' not in instances[0]:
        entityname = 'entity_offsets'

    tag_lists = []
    for instance in instances:
        tags = ['O'] * len(instance['tokens'])
        for entity in instance[entityname]:
            for i in range(entity['offset'][0],entity['offset'][-1]):
                tags[i] = entity['type']
        tag_lists.append(tags)

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

def extract_target_labels(dataset, model, tokenizer):
    data_collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=32)
    vecs = None
    vecs_mu = None
    vecs_sigma = None
    labels = None
    model.eval()
    for batch in tqdm.tqdm(dataloader, desc="Support representations"):
        batch = {k:v.cuda() for k,v in batch.items()}
        label_batch = batch['labels']


        with torch.no_grad():
            inputs = {"input_ids": batch['input_ids'], "attention_mask": batch['attention_mask']}
            outputs = model(**inputs)
            output_embed_mu = outputs[0]
            output_embed_sigma = outputs[1]
            hidden_states = outputs[2]

        if vecs_mu is None:
            vecs = hidden_states.detach().cpu().numpy()
            vecs_mu = output_embed_mu.detach().cpu().numpy()
            vecs_sigma = output_embed_sigma.detach().cpu().numpy()
            labels = label_batch.detach().cpu().numpy()
        else:
            vecs = np.append(vecs, hidden_states.detach().cpu().numpy(), axis=0)
            vecs_mu = np.append(vecs_mu, output_embed_mu.detach().cpu().numpy(), axis=0)
            vecs_sigma = np.append(vecs_sigma, output_embed_sigma.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_batch.detach().cpu().numpy(), axis=0)
    _, _, hidden_size = vecs_mu.shape
    _, _, hidden_bert_size = vecs.shape
    vecs, vecs_mu, vecs_sigma, labels = vecs.reshape(-1, hidden_bert_size), vecs_mu.reshape(-1, hidden_size), vecs_sigma.reshape(-1, hidden_size), labels.reshape(-1)
    fil_vecs, fil_vecs_mu, fil_vecs_sigma, fil_labels = [], [], [], []
    for vec, vec_mu, vec_sigma, label in zip(vecs, vecs_mu, vecs_sigma, labels):
        if label == CrossEntropyLoss().ignore_index:
            continue
        fil_vecs.append(vec)
        fil_vecs_mu.append(vec_mu)
        fil_vecs_sigma.append(vec_sigma)
        fil_labels.append(label)
    vecs, vecs_mu, vecs_sigma, labels = torch.tensor(fil_vecs).cuda(), torch.tensor(fil_vecs_mu).cuda(), torch.Tensor(fil_vecs_sigma).cuda(), torch.tensor(fil_labels).cuda()
    # vecs, vecs_mu, vecs_sigma, labels = torch.tensor(np.array(fil_vecs)), torch.tensor(np.array(fil_vecs_mu)), torch.Tensor(np.array(fil_vecs_sigma)), torch.tensor(np.array(fil_labels))
    return vecs_mu.view(-1, hidden_size), vecs_sigma.view(-1, hidden_size), vecs.view(-1, hidden_bert_size), labels.view(-1)

def evaluate(model, eval_dataloader, labels, sup_dataset, tokenizer):

    model.cuda()

    sup_mus, sup_sigmas, sups, sup_labels = extract_target_labels(sup_dataset, model, tokenizer)

    # Eval!
    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k:v.cuda() for k,v in batch.items()}
        label_batch = batch['labels']
        with torch.no_grad():
            inputs = {"input_ids":batch['input_ids'], "attention_mask": batch['attention_mask']}
            outputs = model(**inputs)
            hidden_states = outputs[2]
            output_embedding_mu = outputs[0]
            output_embedding_sigma = outputs[1]

            nn_predictions, nn_scores = nearest_neighbor(output_embedding_mu, output_embedding_sigma, hidden_states, sup_mus, sup_sigmas, sups, sup_labels, evaluation_criteria='euclidean_hidden_state', num_labels=len(labels))
        if preds is None:
            preds = nn_predictions.detach().cpu().numpy()
            scores = nn_scores.detach().cpu().numpy()
            out_label_ids = label_batch.detach().cpu().numpy()
            
        else:
            preds = np.append(preds, nn_predictions.detach().cpu().numpy(), axis=0)
            scores = np.append(scores, nn_scores.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_batch.detach().cpu().numpy(), axis=0)

    label_map = labels

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    scores_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(out_label_ids[i][j])
                scores_list[i].append(scores[i][j])
                preds_list[i].append(preds[i][j])

    trans_priors = get_abstract_transitions(sup_dataset)
    vit_preds_list = [[] for _ in range(out_label_ids.shape[0])]
    crf = ViterbiDecoder(len(label_map) + 1, trans_priors, 0.01)
    for i in range(out_label_ids.shape[0]):
        sent_scores = torch.tensor(scores_list[i])
        sent_probs = F.softmax(sent_scores/0.0001, dim=1)
        sent_len, n_tag = sent_probs.shape
        feats = crf.forward(torch.log(sent_probs).view(1, sent_len, n_tag))
        vit_tags = crf.viterbi(feats)
        vit_tags = vit_tags.view(sent_len)
        vit_tags = vit_tags.detach().cpu().numpy()
        for tag in vit_tags:
            vit_preds_list[i].append(tag - 1)
    preds_list = vit_preds_list

    return preds_list, out_label_list

def nt_xent(loss, num, denom, temperature = 1):

    loss = torch.exp(loss/temperature)
    cnts = torch.sum(num, dim = 1)
    loss_num = torch.sum(loss * num, dim = 1)
    loss_denom = torch.sum(loss * denom, dim = 1)
    # sanity check
    nonzero_indexes = torch.where(cnts > 0)
    loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

    loss_final = -torch.log2(loss_num) + torch.log2(loss_denom) + torch.log2(cnts)
    return loss_final

def loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    '''
    Calculates KL-divergence between two DIAGONAL Gaussians.
    Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians.
    Note: We calculated both directions of KL-divergence.
    '''
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d

def euclidean_distance(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    logits = ((a - b) ** 2).sum(dim=1)
    return logits


def remove_irrelevant_tokens_for_loss(self, attention_mask, original_embedding_mu, original_embedding_sigma, labels):
    active_indices = attention_mask.view(-1) == 1
    active_indices = torch.where(active_indices == True)[0]

    output_embedding_mu = original_embedding_mu.view(-1, self.embedding_dimension)[active_indices]
    output_embedding_sigma = original_embedding_sigma.view(-1, self.embedding_dimension)[active_indices]
    labels_straightened = labels.view(-1)[active_indices]

    # remove indices with negative labels only

    nonneg_indices = torch.where(labels_straightened >= 0)[0]
    output_embedding_mu = output_embedding_mu[nonneg_indices]
    output_embedding_sigma = output_embedding_sigma[nonneg_indices]
    labels_straightened = labels_straightened[nonneg_indices]

    return output_embedding_mu, output_embedding_sigma, labels_straightened


def calculate_KL_or_euclidean(self, attention_mask, original_embedding_mu, original_embedding_sigma, labels,
                              consider_mutual_O=False, loss_type=None):

    # we will create embedding pairs in following manner
    # filtered_embedding | embedding ||| filtered_labels | labels
    # repeat_interleave |            ||| repeat_interleave |
    #                   | repeat     |||                   | repeat
    # extract only active parts that does not contain any paddings

    output_embedding_mu, output_embedding_sigma, labels_straightened = remove_irrelevant_tokens_for_loss(self, attention_mask,original_embedding_mu, original_embedding_sigma, labels)

    # remove indices with zero labels, that is "O" classes
    if not consider_mutual_O:
        filter_indices = torch.where(labels_straightened > 0)[0]
        filtered_embedding_mu = output_embedding_mu[filter_indices]
        filtered_embedding_sigma = output_embedding_sigma[filter_indices]
        filtered_labels = labels_straightened[filter_indices]
    else:
        filtered_embedding_mu = output_embedding_mu
        filtered_embedding_sigma = output_embedding_sigma
        filtered_labels = labels_straightened

    filtered_instances_nos = len(filtered_labels)

    # repeat interleave
    filtered_embedding_mu = torch.repeat_interleave(filtered_embedding_mu, len(output_embedding_mu), dim=0)
    filtered_embedding_sigma = torch.repeat_interleave(filtered_embedding_sigma, len(output_embedding_sigma),dim=0)
    filtered_labels = torch.repeat_interleave(filtered_labels, len(output_embedding_mu), dim=0)

    # only repeat
    repeated_output_embeddings_mu = output_embedding_mu.repeat(filtered_instances_nos, 1)
    repeated_output_embeddings_sigma = output_embedding_sigma.repeat(filtered_instances_nos, 1)
    repeated_labels = labels_straightened.repeat(filtered_instances_nos)

    # avoid losses with own self
    loss_mask = torch.all(filtered_embedding_mu != repeated_output_embeddings_mu, dim=-1).int()
    loss_weights = (filtered_labels == repeated_labels).int()
    loss_weights = loss_weights * loss_mask

    #ensure that the vector sizes are of filtered_instances_nos * filtered_instances_nos
    assert len(repeated_labels) == (filtered_instances_nos * filtered_instances_nos), "dimension is not of square shape."

    if loss_type == "euclidean":
        loss = -euclidean_distance(filtered_embedding_mu, repeated_output_embeddings_mu, normalize=True)

    elif loss_type == "KL":  # KL_divergence
        loss = -loss_kl(filtered_embedding_mu, filtered_embedding_sigma,
                            repeated_output_embeddings_mu, repeated_output_embeddings_sigma,
                            embed_dimension=self.embedding_dimension)

    else:
        raise Exception("unknown loss")

    # reshape the loss, loss_weight, and loss_mask
    loss = loss.view(filtered_instances_nos, filtered_instances_nos)
    loss_mask = loss_mask.view(filtered_instances_nos, filtered_instances_nos)
    loss_weights = loss_weights.view(filtered_instances_nos, filtered_instances_nos)

    loss_final = nt_xent(loss, loss_weights, loss_mask, temperature = 1)
    return torch.mean(loss_final)


class BertForTokenClassification(BertPreTrainedModel): # modified the original huggingface BertForTokenClassification to incorporate gaussian
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embedding_dimension = config.task_specific_params['embedding_dimension']

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, self.embedding_dimension + (config.hidden_size - self.embedding_dimension) // 2)
        )

        self.output_embedder_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.hidden_size,
                      self.embedding_dimension)
        )

        self.output_embedder_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.hidden_size,
                      self.embedding_dimension)
        )


        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_type='KL',
            consider_mutual_O=True
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = self.dropout(outputs[0])
        original_embedding_mu = ((self.output_embedder_mu((sequence_output))))
        original_embedding_sigma = (F.elu(self.output_embedder_sigma((sequence_output)))) + 1 + 1e-14
        outputs = (original_embedding_mu, original_embedding_sigma,) + (outputs[0],) + outputs[2:]

        if labels is not None:
            loss = calculate_KL_or_euclidean(self, attention_mask, original_embedding_mu,
                                                     original_embedding_sigma, labels, consider_mutual_O,
                                                     loss_type=loss_type)
            outputs = (loss,) + outputs
        return outputs  # (loss), output_mus, output_sigmas, (hidden_states), (attentions)

def _loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    n = mu_i.shape[0]
    m = mu_j.shape[0]

    mu_i = mu_i.unsqueeze(1).expand(n,m, -1)
    sigma_i = sigma_i.unsqueeze(1).expand(n,m,-1)
    mu_j = mu_j.unsqueeze(0).expand(n,m,-1)
    sigma_j = sigma_j.unsqueeze(0).expand(n,m,-1)
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=2)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=2)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d

def entitywise_max(scores, tags, addone=0, num_labels = None):
    # scores: n x m
    # tags: m
    # return: n x t
    n, m = scores.shape
    if num_labels == None:
        max_tag = torch.max(tags) + 1
    else:
        max_tag = num_labels # extra 1 is not needed since it's already 1 based counting
    ret = -100000. * torch.ones(n, max_tag+addone).to(scores.device)
    for t in range(addone, max_tag+addone):
        mask = (tags == (t-addone)).float().view(1, -1)
        masked = scores * mask
        masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
        ret[:, t] = torch.max(masked, dim=1)[0]
    return ret

def _euclidean_metric(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def nearest_neighbor(rep_mus, rep_sigmas, rep_hidden_states, support_rep_mus, support_rep_sigmas, support_rep, support_tags, evaluation_criteria, num_labels):
    """
    Neariest neighbor decoder for the best named entity tag sequences
    """
    batch_size, sent_len, ndim = rep_mus.shape
    _, _, ndim_bert = rep_hidden_states.shape

    # scores = _loss_kl(rep_mus.view(-1, ndim), rep_sigmas.view(-1,ndim), support_rep_mus, support_rep_sigmas, ndim)
    # tags = support_tags[torch.argmin(scores, 1)]
    scores = _euclidean_metric(rep_hidden_states.view(-1, ndim_bert), support_rep, True)
    tags = support_tags[torch.argmax(scores, 1)]

    scores = entitywise_max(scores, support_tags, 1, num_labels)
    max_scores, tags = torch.max(scores, 1)
    tags = tags - 1

    return tags.view(batch_size, sent_len), scores.view(batch_size, sent_len, -1)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    previous_word_idx = None
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(-100)
        previous_word_idx = word_idx
    return new_labels

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def labelpad(self, labels, features):
        sequence_length = len(features["input_ids"][0])
        features["labels"] = [
            list(label) + [-100] * (sequence_length - len(label)) for label in labels
        ]
        return features
    
    def getpad(self,features,labels=None,labelmask=None):
        features = self.tokenizer.pad(
                features,
                padding=True,
            )
        if labels is not None:
            features = self.labelpad(labels,features)
        return features

    def __call__(self, features):

        rawinputs = []
        rawlabels = []
        for feature in features:
            rawfeature = {}
            rawfeature['input_ids'] = feature['input_ids']
            rawfeature['attention_mask'] = feature['attention_mask']
            rawinputs.append(rawfeature)
            rawlabels.append(feature['labels'])
        rawinputs = self.getpad(rawinputs,labels=rawlabels)
        inputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in rawinputs.items()}
        return inputs

class ContainerDataset(Dataset):
    def __init__(self, dataset, tokenizer, label2id):
        super(ContainerDataset).__init__()
        self.dataset = dataset
        self.label2id = label2id
        self.tokenizer = tokenizer
    
    def getdata(self,instance):
        tokens = instance['tokens']
        tokenized_inputs = self.tokenizer(tokens,max_length=200, padding='max_length', truncation=True, is_split_into_words=True)
        new_labels = []
        labels = [0] * len(tokens)
        for entity in instance:
            for entity in instance['entity']:
                for index in range(entity['offset'][0],entity['offset'][-1]):
                    labels[index] = self.label2id[entity['type']]
        word_ids = tokenized_inputs.word_ids(0)
        new_labels = align_labels_with_tokens(labels,word_ids)
        tokenized_inputs["labels"] = new_labels
        tokenized_inputs['entity'] = instance['entity']
        tokenized_inputs['tokens'] = instance['tokens']
        return tokenized_inputs
    
    def __getitem__(self, index):
        return self.getdata(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)



def tag2entity(id2label,pred,gold,tokens):
    entitys = []
    lastlabel = 'O'
    nowlabel = 'O'
    start = -1
    end = -1
    index = 0
    rawlabels = []
    for i in range(gold.shape[0]):
        if gold[i] >= 0:
            label = id2label[pred[i]]
            rawlabels.append(label)
            if label != 'O':
                if lastlabel == label:
                    entity.append(tokens[index])
                    end = index
                else:
                    if nowlabel != 'O':
                        entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
                        nowlabel = 'O'
                    entity = [tokens[index]]
                    start = index
                    end = index
                    lastlabel = label
                    nowlabel = label
            else:
                if nowlabel != 'O':
                    entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
                    nowlabel = 'O'
                entity = []
                start = -1
                end = -1
                lastlabel = 'O'
            index += 1
    if nowlabel != 'O':
        entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
        nowlabel = 'O'
    return entitys,rawlabels



class Model:
    def __init__(self, model, tokenizer, dataset, targetlabels, training_args, support_dataset = None):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollator(tokenizer=tokenizer)
        self.targetlabels = targetlabels
        self.label2id = {'O':0}
        for label in targetlabels:
            self.label2id[label] = len(self.label2id)
        self.dataset = dataset
        self.processed_dataset = ContainerDataset(dataset,tokenizer,self.label2id)
        if support_dataset is not None:
            self.support_dataset = ContainerDataset(support_dataset,tokenizer,self.label2id)
        self.training_args = training_args
        self.id2label = {}
        for label in self.label2id:
            self.id2label[len(self.id2label)] = label

    def tag2entity(self, id2label,pred,gold,tokens):
        entitys = []
        lastlabel = 'O'
        nowlabel = 'O'
        start = -1
        end = -1
        index = 0
        rawlabels = []
        for i in range(gold.shape[0]):
            if gold[i] >= 0:
                label = id2label[pred[i]]
                rawlabels.append(label)
                if label != 'O':
                    if lastlabel == label:
                        entity.append(tokens[index])
                        end = index
                    else:
                        if nowlabel != 'O':
                            entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
                            nowlabel = 'O'
                        entity = [tokens[index]]
                        start = index
                        end = index
                        lastlabel = label
                        nowlabel = label
                
                else:
                    if nowlabel != 'O':
                        entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
                        nowlabel = 'O'
                    entity = []
                    start = -1
                    end = -1
                    lastlabel = 'O'
                index += 1
            
        if nowlabel != 'O':
            entitys.append({'type':nowlabel, 'offset':[start,end+1], 'text':' '.join(entity)})
            nowlabel = 'O'
        return entitys,rawlabels
    
    def finetune(self):
        self.model = self.model.cuda()

        previous_score = 1e+6 # infinity placeholder
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)
        # Train!

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        rep_index = -1

        dataloader = DataLoader(self.processed_dataset, collate_fn=self.data_collator, batch_size=self.training_args.per_device_train_batch_size)

        while(True):
            rep_index += 1
            epoch_iterator = tqdm.tqdm(dataloader, desc="Iteration", disable=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                # here loss can be either KL, or euclidean.
                batch = {k:v.cuda() for k,v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs[0]
                # logger.info("finetune loss at repetition "+ str(rep_index) + " : " + str(loss.item()))
                loss.backward()
                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                optimizer.step()
                self.model.zero_grad()

            if loss.item() > previous_score:
                # early stopping with single step patience
                break

            previous_score = loss.item()

        output_dir = self.training_args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.training_args, os.path.join(output_dir, "training_args.bin"))

    def predict(self):
        
        eval_dataloader = DataLoader(self.processed_dataset, collate_fn=self.data_collator, batch_size=self.training_args.per_device_eval_batch_size)

        predictions, labels = evaluate(self.model, eval_dataloader, self.id2label, self.support_dataset, self.tokenizer)

        pred_entities = []
        index = 0
        for label, predict in zip(labels, predictions):
            predict = np.array(predict)
            label = np.array(label)
            pred_entities.append(tag2entity(self.id2label,predict,label,self.dataset[index]['tokens']))
            index += 1

        # del model
        gc.collect()
        torch.cuda.empty_cache()
        with open(self.training_args.output_dir + '/prediction.json', 'w') as f:
            for index in range(len(pred_entities)):
                instance = self.dataset[index]
                if 'entity' not in instance:
                    instance['entity'] = instance['entity_offsets']
                f.write(json.dumps({'index':index,'gold': instance['entity'], 'pred': pred_entities[index][0], 'targetlabels': self.id2label})+'\n')
        return pred_entities
