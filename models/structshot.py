import json
import logging
import os
from transformers import Trainer
import evaluate
import numpy as np
import torch
import torch.nn as nn
import random 
import evaluate
import sys
import copy
import torch.nn.functional as F


from typing import Optional
import gc
import tqdm 

import json
import logging
import os
from transformers import AutoModel,TrainingArguments,Trainer,HfArgumentParser,AutoTokenizer, set_seed
import evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset
import random 
import evaluate
import sys
import torch.nn.functional as F



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

def nn_decode(reps, support_reps, support_tags):
    """
    NNShot: neariest neighbor decoder for few-shot NER from https://github.com/asappresearch/structshot/blob/2bf53794b3ffd55b9970eb7e3c4b68847b1bd4eb/structshot/run_pl_pred.py
    """
    batch_size, sent_len, ndim = reps.shape
    scores = _euclidean_metric(reps.view(-1, ndim), support_reps.view(-1,ndim), True)
    # tags = support_tags[torch.argmax(scores, 1)]
    emissions = get_nn_emissions(scores, support_tags)
    tags = torch.argmax(emissions, 1)
    return tags.view(batch_size, sent_len), emissions.view(batch_size, sent_len, -1)

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

def get_nn_emissions(scores, tags):
    """
    Obtain emission scores from NNShot
    """
    n, m = scores.shape
    n_tags = torch.max(tags) + 1
    emissions = -100000. * torch.ones(n, n_tags).to(scores.device)
    for t in range(n_tags):
        mask = (tags == t).float().view(1, -1)
        masked = scores * mask
        masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
        emissions[:, t] = torch.max(masked, dim=1)[0]
    return emissions


def getevallogits(model,inputs, support_feature, support_labels):
    # n,d

    info = inputs['info']
    n = info['n']
    
    query = {
        'input_ids': inputs['query_input_ids'],
        'attention_mask': inputs['query_attention_mask']
    }
    query_outputs = model(**query)
    query_feature = query_outputs.get("last_hidden_state") # q,t,d
    pred, logits = nn_decode(query_feature, support_feature,support_labels)
    return logits


class NNshotTrainer(Trainer):
    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys,
        ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.support_feature is None:
                self.support_feature, self.support_labels = self.get_support_encodings_and_labels(model, inputs)
            labels = inputs.get("labels")
            logits = getevallogits(model,inputs, self.support_feature, self.support_labels)
            loss = None
        return (loss, logits, labels)
    
    
    def get_support_encodings_and_labels(self, model, inputs):
        """
        Get token encodings and labels for all tokens in the support set
        """
        support = {
            'input_ids': inputs['support_input_ids'],
            'attention_mask': inputs['support_attention_mask']
        }
        support_outputs = model(**support)
        support_reps = support_outputs.get("last_hidden_state")
        support_tags = inputs['support_labels']
        support_encodings, support_labels = [], []
        batchsize = support_reps.size(0)
        for batch in range(batchsize):
            encodings = support_reps[batch].view(-1, support_reps.size(-1))
            labels = support_tags[batch]
            labels = labels.flatten()
            # filter out PAD tokens
            idx = torch.where(labels != -100)[0]
            support_encodings.append(encodings[idx])
            support_labels.append(labels[idx])
        return torch.cat(support_encodings), torch.cat(support_labels)

class protoEvalDataset(Dataset):
    def __init__(self, dataset, trainset, targetlabels = None):
        super(protoEvalDataset).__init__()
        self.dataset = dataset
        self.train = trainset
        self.targetlabels = targetlabels
    
    def __getitem__(self, index):
        return self.train, [self.dataset[index]], self.targetlabels, {'n': len(self.targetlabels), 'k': len(self.train)/len(self.targetlabels)}
    
    def __len__(self):
        return len(self.dataset)

class DataCollatorForProto:
    def __init__(self,tokenizer,padding=True,max_length=128,label_pad_token_id=-100) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length  = max_length
        self.label_pad_token_id = label_pad_token_id
    
    def align_labels_with_tokens(self,labels, word_ids):
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

    def getexamples(self,instances):
        tokens = []
        entities = []
        for instance in instances:
            tokens.append(instance['tokens'])
            if 'entity' not in instance:
                entities.append(instance['entity_offsets'])
            else:
                entities.append(instance['entity'])
        return {'tokens':tokens,'entity':entities}

    def tokenize_and_align_labels(self,instances,label2id):
        examples = self.getexamples(instances)
        tokenized_inputs = self.tokenizer(
            examples["tokens"], max_length=self.max_length, truncation=True, is_split_into_words=True
        )
        new_labels = []
        for i,entities in enumerate(examples['entity']):
            labels = [0] * len(examples['tokens'][i])
            for entity in entities:
                for index in range(entity['offset'][0],entity['offset'][-1]):
                    if index >= len(labels):
                        break
                    if entity['type'] in label2id:
                        labels[index] = label2id[entity['type']]
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def labelpadding(self,instances):
        sequence_length = len(instances["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            instances["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in instances['labels']
            ]
        else:
            instances["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in instances['labels']
            ]
        return instances

    def __call__(self, data):

        support_instances_full = {}
        query_instances_full = {}

        for eachdata in data:
            info = eachdata[3]
            labels = eachdata[2]
            support_instances = eachdata[0]
            query_instances = eachdata[1]
            label2id = {}
            for label in labels:
                label2id[label] = len(label2id) + 1
            support_instances = self.tokenize_and_align_labels(support_instances,label2id)
            query_instances = self.tokenize_and_align_labels(query_instances,label2id)
            for key in support_instances:
                if key not in support_instances_full:
                    support_instances_full[key] = []
                support_instances_full[key] += support_instances[key]
            for key in query_instances:
                if key not in query_instances_full:
                    query_instances_full[key] = []
                query_instances_full[key] += query_instances[key]
        support_instances = self.tokenizer.pad(
            support_instances_full,
            padding=self.padding,
        )
        query_instances = self.tokenizer.pad(
            query_instances_full,
            padding=self.padding,
        )
        
        support_instances = self.labelpadding(support_instances)
        query_instances = self.labelpadding(query_instances)


        features = {
            'support_input_ids': support_instances['input_ids'],
            'support_attention_mask': support_instances['attention_mask'],
            'support_labels': support_instances['labels'],
            'query_input_ids': query_instances['input_ids'],
            'query_attention_mask': query_instances['attention_mask'],
            'labels': query_instances['labels'],
        }

    

        features = {k: torch.tensor(v, dtype=torch.int64) for k, v in features.items()}

        features['info'] = info

        return features




import torch
import torch.nn as nn


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

def structshot(trainset,targets, out_label_ids, emissions_list):
    abstract_transitions = get_abstract_transitions(trainset)
    viterbi_decoder = ViterbiDecoder(len(targets)+1, abstract_transitions, 0.5)
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in tqdm.tqdm(range(out_label_ids.shape[0])):
        sent_scores = torch.tensor(emissions_list[i])
        sent_len, n_label = sent_scores.shape
        sent_probs = F.softmax(sent_scores/0.0001, dim=1)
        start_probs = torch.zeros(sent_len) + 1e-6
        sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
        feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
        vit_labels = viterbi_decoder.viterbi(feats)
        vit_labels = vit_labels.view(sent_len)
        vit_labels = vit_labels.detach().cpu().numpy()
        # print(vit_labels)
        for label in vit_labels:
            if label > 0:
                preds_list[i].append(label-1)
            else:
                preds_list[i].append(label)
        # raise
    return preds_list
        

class Model:
    def __init__(self, model, tokenizer, dataset, targetlabels, training_args, support_dataset = None):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForProto(tokenizer=tokenizer)
        self.targetlabels = targetlabels
        self.label2id = {'O':0}
        for label in targetlabels:
            self.label2id[label] = len(self.label2id)
        self.dataset = dataset
        self.support_dataset = support_dataset
        self.processed_dataset = protoEvalDataset(dataset,support_dataset,self.targetlabels)
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
    
    def predict(self):
        self.training_args.remove_unused_columns = False
        trainer = NNshotTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.support_feature = None
        result = trainer.predict(self.processed_dataset)
        predictions = result.predictions
        predictions = np.argmax(predictions,axis=-1)
        labels = result.label_ids

        emissions_list = [[] for _ in range(predictions.shape[0])]
        
        newlabel = [[] for _ in range(predictions.shape[0])]
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    emissions_list[i].append(result.predictions[i][j])
                    newlabel[i].append(labels[i][j])
        emissions_list = [np.array(i) for i in emissions_list]
        predictions = structshot(self.support_dataset, self.id2label, predictions, emissions_list)
        labels = [np.array(i) for i in newlabel]

        pred_entities = []
        index = 0
        for label, predict in zip(labels,predictions):
            pred_entities.append(self.tag2entity(self.id2label,predict,label,self.processed_dataset[index][1][0]['tokens']))
            index += 1

        gc.collect()
        torch.cuda.empty_cache()
        savepath = self.training_args.output_dir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        with open(savepath + '/prediction.json', 'w') as f:
            for index in range(len(pred_entities)):
                instance = self.processed_dataset[index][1][0]
                f.write(json.dumps({'index':index,'gold': instance['entity'], 'pred': pred_entities[index][0], 'targetlabels': self.id2label})+'\n')
        return pred_entities