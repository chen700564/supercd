import math
import tqdm 
import torch
import json
import sys
import random
import gc
import os
import numpy as np
import copy
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def inner(a,b):
    return np.matmul(a,b.T)

def sent_to_instance(tokenizer,dataset):
    instances = []
    for data in dataset:
        text = ' '.join(data['tokens'])
        sent_id = tokenizer.encode('[-Text] ' +text,max_length=128,add_special_tokens=False)
        mask = [1] * len(sent_id)
        instance = {
            'input_ids': sent_id,
            'attention_mask': mask,
        }
        instances.append(instance)
    return instances

def query_to_instance(tokenizer,dataset):
    instances = []
    for data in dataset:
        query = ''
        removed = data['removed']
        concepts = data['concepts']
        for concept in concepts:
            query = query + concept + ' [-,]'

        query = '[-Query] [-R] ' + removed + ' [-Q] ' + query

        sent_id = tokenizer.encode(query,max_length=128,add_special_tokens=False)
        mask = [1] * len(sent_id)
        instance = {
            'input_ids': sent_id,
            'attention_mask': mask,
        }
        instances.append(instance)
    return instances

def getsentfeat(dataset,model,tokenizer,query=False):
    model.cuda()
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if query:
        dataset = query_to_instance(tokenizer, dataset)
    else:
        dataset = sent_to_instance(tokenizer,dataset)

    dataloader = DataLoader(dataset, batch_size=256, collate_fn=data_collator)

    sent_feats = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {k:v.to(model.device) for k,v in batch.items()}
            outputs = model(**batch)
            feats = outputs.last_hidden_state[:,0].detach().cpu().numpy()
            sent_feats = sent_feats + [i for i in feats]
    sent_feats = np.array(sent_feats)

    return sent_feats




def retrival(dataset, sent_feats, concept_feats, concepts,modelfile):
    results = {}

    for label in concepts:

        results[label] = {}
        if len(concepts[label]) > 0:     

            scores = inner(concept_feats[label],sent_feats)

            sortedindexs = np.argsort(-np.array(scores),axis=-1)
            sortedscores = -np.sort(-np.array(scores),axis=-1)

            conceptindex = list(range(len(concepts[label])))
        else:
            conceptindex = list(range(len(concepts[label])))
        with open(modelfile + '/retrival_'+label.replace('/','_')+'.json','w') as f:
            for index in conceptindex:
                removed = concepts[label][index]['removed']
                concepttype = concepts[label][index]['type']
                sortedindex = sortedindexs[index].tolist()
                sortedscore = sortedscores[index].tolist()
                results[label][removed] = [[sortedindex[i],sortedscore[i],dataset[sortedindex[i]]['index'],concepttype] for i in range(len(sortedindex))]
                f.write(json.dumps({'removed':removed,'result':results[label][removed], 'type': concepttype})+'\n')
    return results

def getsample(datasets,results,entities,repeat=False, num =500):
    index = {}
    labelindex = {}
    for label in results:
        labelindex[label] = 0
        index[label] = []
        for concept in results[label]:
            index[label] += [[i[0],idx,concept,i[-1]] for idx,i in  enumerate(results[label][concept][:num]) if i[1] > 0] 
        index[label] = sorted(index[label],key=lambda k:k[1])
    indexs = []
    usedtokens = []
    for idx in range(num):
        for label in index:
            flag = True
            cnum = 1
            for j in range(cnum):
                while flag:
                    if len(index[label]) <= labelindex[label]:
                        break
                    sentindex = index[label][labelindex[label]][0]
                    removed = index[label][labelindex[label]][2]
                    tokens = datasets[sentindex]['tokens']
                    if sentindex not in indexs and tokens not in usedtokens and len(tokens) > 5:

                        flag2 = True
                        texts = []
                        for entity in datasets[sentindex]['entity']:
                            if entity['type'] == label:
                                text = entity['text'].lower()
                                if text not in entities[label]:
                                    texts.append(text)
                                else:
                                    flag2 = False
                                    break
                        if not repeat or flag2:
                            entities[label] += texts
                            indexs.append([index[label][labelindex[label]][0],label,removed])
                            usedtokens.append(tokens)
                            flag = False
                    labelindex[label] += 1
    picked = []

    for idx in indexs:
        newinstance = copy.deepcopy(datasets[idx[0]])
        newinstance['label'] = idx[1]
        newinstance['removed'] = idx[2]
        picked.append(newinstance)
    return picked

def conceptobaining(supportresult,concept_mapping,maxrate,withfilter=True):
    vocab = []
    stats = {}
    for label in concept_mapping:
        stats[label] = {'freqs':[],'num':0,'related_concepts':{}}
    for sentresult in supportresult:
        for entityresult in sentresult:
            predconcepts = entityresult[1].split(',')
            concepts = []
            for i in predconcepts:
                i = i.strip()
                if i not in concepts:
                    concepts.append(i)
            
            stats[entityresult[0][1]]['num'] += 1
            for concept in concepts:
                if concept == 'other thing':
                    continue

                if concept not in stats[entityresult[0][1]]['related_concepts']:
                    stats[entityresult[0][1]]['related_concepts'][concept] = copy.deepcopy(concepts)
                else:
                    for i in concepts:
                        if i not in stats[entityresult[0][1]]['related_concepts'][concept]:
                            stats[entityresult[0][1]]['related_concepts'][concept].append(i)
                
                if concept not in vocab:
                    vocab.append(concept)
                    for label in stats:
                        if label == entityresult[0][1]:
                            stats[label]['freqs'].append(1)
                        else:
                            stats[label]['freqs'].append(0)
                else:
                    conceptid = vocab.index(concept)
                    stats[entityresult[0][1]]['freqs'][conceptid] += 1
                
    inter_common = {}
    for label in stats:
        common,unique = classifyconcepts(stats[label]['freqs'],stats[label]['num'],vocab,max_rate=maxrate)
        for concept in common:
            if concept not in inter_common:
                inter_common[concept] = 1
            else:
                inter_common[concept] += 1
        concept_mapping[label]['common'] = common
        concept_mapping[label]['unique'] = unique

    full_common = [i for i in inter_common]
    inter_common = [i for i in inter_common if inter_common[i] > 1]
    

    for label in concept_mapping:
        common = concept_mapping[label]['common']
        unique = concept_mapping[label]['unique']
        concept_mapping[label]['inter-common'] = [i for i in common if i in inter_common]
        concept_mapping[label]['inter-common-unique'] = [i for i in unique if i in full_common]
        concept_mapping[label]['common'] = [i for i in common if i not in inter_common]
        if withfilter and len(common) > 0:
            concept_mapping[label]['unique'] = []
            for concept in unique:
                if concept in full_common:
                    continue
                related = stats[label]['related_concepts'][concept]
                flag = False
                for commonconcept in common:
                    if commonconcept in related:
                        flag = True
                        break
                if flag:
                    concept_mapping[label]['unique'].append(concept)
        else:
            concept_mapping[label]['unique'] = [i for i in unique if i not in full_common]
        concept_mapping[label]['full'] = common + unique
        concept_mapping[label]['concepts'] = stats[label]['related_concepts']
        concept_mapping[label]['freq'] = {}
        for concept in concept_mapping[label]['common']+concept_mapping[label]['unique']:
            index = vocab.index(concept)
            concept_mapping[label]['freq'][concept] = stats[label]['freqs'][index]
    return concept_mapping

def activeconceptpicker(supportresult,concept_mapping, baned):

    stats = {}
    for label in concept_mapping:
        stats[label] = {}
    for sentresult in supportresult:
        for entityresult in sentresult:
            predconcepts = entityresult[1].split(',')
            label = entityresult[0][1]
            for concept in predconcepts:
                concept = concept.strip()
                if concept == 'other thing':
                    continue
                if len(concept) == 0:
                    continue
                if concept not in stats[label]:
                    stats[label][concept] = {'num': 1, 'related_concepts':[]}
                else:
                    stats[label][concept]['num'] += 1
                
                for i in predconcepts:
                    i = i.strip()
                    if len(i) == 0 or i == concept or i in stats[label][concept]['related_concepts']:
                        continue
                    stats[label][concept]['related_concepts'].append(i)
    

    accepted_labels = {}
    labelscores = {}
    scores = {}
    for label in stats:
        accepted_labels[label] = []
        labelscores[label] = []
        scores[label] = {}
        for concept in stats[label]:
            if stats[label][concept]['num'] > 1:
                accepted_labels[label].append(concept)
            else:
                flag = False
                for otherconcept in stats[label][concept]['related_concepts']:
                    if stats[label][otherconcept]['num'] > 1:
                        flag = True
                        break
                if flag:
                    accepted_labels[label].append(concept)

    for label in accepted_labels:
        for concept in accepted_labels[label]:
            score = stats[label][concept]['num']
            maxotherscore = 0
            for otherlabel in stats:
                if otherlabel != label and concept in stats[otherlabel]:
                    if stats[otherlabel][concept]['num'] > maxotherscore:
                        maxotherscore = stats[otherlabel][concept]['num']
            labelscores[label].append(score)
            score = score - maxotherscore
            scores[label][concept] = score
    

    sorted_accepted_labels = {}
    for label in labelscores:
        sorted_accepted_labels[label] = [accepted_labels[label][i] for i in np.argsort(np.array(labelscores[label]))]
    concepts = {}
    for label in sorted_accepted_labels:
        concepts[label] = []
        index = 0
        for concept in sorted_accepted_labels[label]:
            if concept not in baned and scores[label][concept] > 0:
                related_concepts = [i for i in stats[label][concept]['related_concepts'] if i not in baned]
                related_concepts = random.sample(related_concepts,min(20,len(related_concepts)))
                concepts[label].append({'index': index, 'removed':concept,'concepts': related_concepts,'type':'picked', 'score': scores[label][concept]})
                index += 1
    return concepts
            

def getsample2(datasets,results,entities,repeat=False):
    index = {}
    num = 500
    labelindex = {}
    sortedresults = {}
    for label in results:
        sortedresults[label] = {}
        concepts = list(results[label].keys())
        scores = [results[label][concept][0][1] for concept in concepts]
        sortedindex = np.argsort(-np.array(scores))
        for i in sortedindex:
            sortedresults[label][concepts[i]] = results[label][concepts[i]]
    results = sortedresults

    for label in results:
        labelindex[label] = 0
        index[label] = []
        for concept in results[label]:
            index[label] += [[i[0],idx,concept,i[-1]] for idx,i in  enumerate(results[label][concept][:num]) if i[1] > 0] 
        index[label] = sorted(index[label],key=lambda k:k[1])
    indexs = []
    usedtokens = []
    for idx in range(num):
        for label in index:
            flag = True
            cnum = 1
            for j in range(cnum):
                while flag:
                    if len(index[label]) <= labelindex[label]:
                        break
                    sentindex = index[label][labelindex[label]][0]
                    removed = index[label][labelindex[label]][2]
                    tokens = datasets[sentindex]['tokens']
                    if sentindex not in indexs and tokens not in usedtokens and len(tokens) > 5:

                        flag2 = True
                        texts = []
                        for entity in datasets[sentindex]['entity']:
                            if entity['type'] == label:
                                text = entity['text'].lower()
                                if text not in entities[label]:
                                    texts.append(text)
                                else:
                                    flag2 = False
                                    break
                        if not repeat or flag2:
                            entities[label] += texts
                            indexs.append([index[label][labelindex[label]][0],label,removed])
                            usedtokens.append(tokens)
                            flag = False
                    labelindex[label] += 1
    picked = []

    for idx in indexs:
        newinstance = copy.deepcopy(datasets[idx[0]])
        newinstance['label'] = idx[1]
        newinstance['removed'] = idx[2]
        picked.append(newinstance)
    return picked





def classifyconcepts(freqs,num,vocab,max_rate=0.5):
    freqs = np.array(freqs)
    common_index = np.where(freqs > max_rate)[0]
    unique_index = np.where(freqs <= max_rate)[0]
    common = [vocab[i] for i in common_index if freqs[i] > 0]
    unique = [vocab[i] for i in unique_index if freqs[i] > 0]
    return common,unique


def getconcepts(concept_mapping,baned=None,inter=False):
    result = {}
    index = 0
    for label in concept_mapping:
        result[label] = []
        if inter:
            common = concept_mapping[label]['common'] + concept_mapping[label]['inter-common']
        else:
            common = concept_mapping[label]['common']
        if len(common) > 0:
            for concept in common:
                if baned is None or concept not in baned:
                    removed = concept
                    concepts = copy.deepcopy(concept_mapping[label]['concepts'][removed])
                    if removed in concepts:
                        concepts.remove(removed)
                    concepts = random.sample(concepts,min(20,len(concepts)))
                    result[label].append({'index':index,'removed':removed,'concepts':concepts,'type':'common','freq':concept_mapping[label]['freq'][removed]})
                    index += 1
        if len(concept_mapping[label]['unique']) > 0:
            for concept in concept_mapping[label]['unique']:
                if baned is None or concept not in baned:
                    removed = concept
                    concepts = copy.deepcopy(concept_mapping[label]['concepts'][removed])
                    if removed in concepts:
                        concepts.remove(removed)
                    concepts = random.sample(concepts,min(20,len(concepts)))
                    result[label].append({'index':index,'removed':removed,'concepts':concepts,'type':'unique','freq':concept_mapping[label]['freq'][removed]})
                    index += 1
    return result

def getscore(query,concepts):
    num = 0
    num2 = 0
    for concept in concepts:
        if concept in query:
            num += 1
    return num


class Sampling:
    def __init__(self,model, modelpath, way):
        self.ce = model[0]
        self.sir = model[1]
        self.modelpath = modelpath
        self.way = way
    
    def getconcept(self):
        return self.ce.conceptgeneration(single = True)

    def sampling(self, unlabeledindex, datasets, picknum):
        supportresult = self.getconcept()
        with open(self.modelpath + '/support_concepts.json','w') as f:
            for j in supportresult:
                    f.write(json.dumps(j)+'\n')
        
        targetlabel = self.ce.targetlabels
        concept_mapping = {}
        for label in targetlabel:
            concept_mapping[label] = {
                'common': [],
                'unique': [],
                'inter-common': []
            }
        concept_mapping = conceptobaining(supportresult,concept_mapping,self.sir.maxrate,True)
        json.dump(concept_mapping,open(self.modelpath + '/concept_mapping.json','w'))
        concepts = getconcepts(concept_mapping,self.sir.baned)
        json.dump(concepts,open(self.modelpath + '/concepts.json','w') )

        sent_feats = getsentfeat(datasets,self.sir,self.sir.tokenizer)
        concept_feats = {}
        for label in concepts:
            concept_feats[label] = getsentfeat(concepts[label],self.sir,self.sir.tokenizer,True)

        unlabeleddata = copy.deepcopy([datasets[i] for i in unlabeledindex])
        for i in range(len(unlabeleddata)):
            unlabeleddata[i]['index'] = unlabeledindex[i]
        unlabeledfeat = np.array([sent_feats[i] for i in unlabeledindex])

        entities = {}
        for label in self.ce.targetlabels:
            entities[label] = []
        for index,instance in enumerate(self.ce.dataset):
            typeset = []
            for entity in instance['entity']:
                if entity['type'] not in typeset and entity['type'] in targetlabel:
                    typeset.append(entity['type'])
                if entity['type'] in entities and entity['text'].lower() not in entities[entity['type']]:
                    entities[entity['type']].append(entity['text'].lower())
        

        results = retrival(unlabeleddata, unlabeledfeat, concept_feats, concepts, self.modelpath)
        additiondatas = getsample(unlabeleddata,results,entities, picknum * 2)


        result = []
        addedtokens = []

        for line in additiondatas:
            if len(line['tokens']) <= 5:
                continue
            if line['tokens'] not in addedtokens:
                addedtokens.append(line['tokens'])
            else:
                continue
            result.append(line)
            if len(result) == picknum:
                break
    
        with open(self.modelpath + '/result.json','w') as f:
            for j in additiondatas:
                    f.write(json.dumps(j)+'\n')

        newresult = []    
        for instance in result:
            newresult.append(datasets[instance['index']])

        return newresult
    
