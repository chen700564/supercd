import os
import logging
import json
from transformers import Seq2SeqTrainer
import numpy as np
import torch,tqdm,copy
import gc, random, math
from torch.utils.data import DataLoader, Dataset
logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)




def obtainwordscore(spanmaps,mapping,other='other thing'):
    id2label = [i for i in mapping.keys()]
    voacb = []
    wordscore = []
    othernum = [0] * len(id2label)
    for spanmap in spanmaps:
        for i in spanmap:
            finelabels = i[1].split(',')
            typeindex = id2label.index(i[0][1])
            for label in finelabels:
                label = label.strip()
                if label == other:
                    othernum[typeindex] += 1
                if len(label) > 0 and label != other:
                    if label not in voacb:
                        voacb.append(label)
                        wordscore.append([0] * len(id2label))
                    wordindex = voacb.index(label)
                    wordscore[wordindex][typeindex] += 1
    othernum = wordscore + [othernum]
    wordscore = np.array(wordscore)
    othernum = np.array(othernum)
    return id2label,voacb,wordscore,othernum

def filterchild(spanmaps,mapping):
    id2label,voacb,wordscore,othernum = obtainwordscore(spanmaps,mapping)
    score1 = copy.deepcopy(wordscore)
    score1 = score1/(np.sum(score1,axis=0)+1e-16)
    score2 = copy.deepcopy(wordscore)
    score2 = score2/(np.sum(score2,axis=1,keepdims=True)+1e-16)
    otherscore = othernum/(np.sum(othernum,axis=0)+1e-16)
    allscore = score1 * score2
    nums = np.sum(copy.deepcopy(wordscore),axis=0)
    spanlabel = np.argmax(allscore,axis=-1)
    spanlabel = [id2label[i] for i in spanlabel]
    for spanmap in spanmaps:
        for i in spanmap:
            finelabels = i[1].split(',')
            labelpred = []
            for label in finelabels:
                label = label.strip()
                if len(label) > 0 and label != 'other thing':
                    typeindex = id2label.index(i[0][1])
                    wordindex = voacb.index(label)
                    value = [label,allscore[wordindex][typeindex]]
                    labelpred.append([wordindex,spanlabel[wordindex]])
            for label in labelpred:
                value = [voacb[label[0]],allscore[label[0]][typeindex]]
                if value not in mapping[i[0][1]]:
                    mapping[i[0][1]].append(value)
    for label in mapping:
        mapping[label] = sorted(mapping[label],key=lambda k:k[1],reverse=True)
        if len(mapping[label]) > 0:
            mapping[label] = [i[0] for i in mapping[label]]
            if otherscore[-1][id2label.index(label)] > 0.5:
                mapping[label] = []
    return mapping


class DataCollator:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.return_tensors = 'pt'
        self.label_pad_token_id = -100
    

    def __call__(self, inputs, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if 'decoder_input_ids' in inputs[0]:
            features = {
                'input_ids': [i['input_ids'] for i in inputs],
                'attention_mask': [i['attention_mask'] for i in inputs],
            }
            features2 = {
                'input_ids': [i['decoder_input_ids'] for i in inputs],
                'attention_mask': [i['decoder_attention_mask'] for i in inputs],
            }
        else:
            features = {
                'input_ids': [i['input_ids'] for i in inputs],
                'attention_mask': [i['attention_mask'] for i in inputs],
            }
            features2 = None
        features = self.tokenizer.pad(
            features,
            padding=True,
        )
        if features2 is not None:
            features2 = self.tokenizer.pad(
                features2,
                padding=True,
            )
            features['decoder_input_ids'] = features2['input_ids']
            features['decoder_attention_mask'] = features2['attention_mask']
            labels = [i['labels'] for i in inputs]
            sequence_length = len(features["decoder_input_ids"][0])
            features["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            features["labels"] = np.array(features["labels"])
        features = {k: torch.tensor(v) for k, v in features.items()}
        return features


class SDNetdataset(Dataset):
    def __init__(self, dataset, mapping, tokenizer, pred = False, typing = False, selfdesb = None, samples = None):
        super().__init__()
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.pred = pred
        self.typing = typing
        self.dataset = dataset
        self.selfdesb = selfdesb
        self.samples = samples    
        if not typing and selfdesb is not None:
            self.dataset = self.getsdnetdataset()
    
    def getsdnetdataset(self):
        dataset = []
        for i in range(len(self.dataset)):
            for singlelabels in self.selfdesb:
                newinstance = copy.deepcopy(self.dataset[i])
                newinstance['index'] = i
                newinstance['labels'] = singlelabels   
                dataset.append(newinstance)
        return dataset
    
    def entity2text(self,entitys,labelmap=None, targets=None):
        generated = []
        for entity in entitys:
            type = entity['type']
            type = labelmap[type]
            if targets is not None and entity['type'] not in targets:
                continue
            generated.append( entity['text'] + ' is' + ' ' + type + '.')
        return generated

    def prefix_generator(self,instances,typing=False,labelmap=None):
        '''
        instances = labels / spans
        labels: [[Label1,[finelabel1,finelabel2],[Label2,[finelabel3,finelabel4]]
        spans: [span1,span2]
        FET: {Typing | &span1 &span2}
        NER: {NER | & Label1: finelabel1, finelabel2 & Label2: finelabel3, finelabel4}
        { <extra_id_0> 32099
        } <extra_id_1> 32098
        | <extra_id_2> 32097
        : <extra_id_3> 32096
        , <extra_id_4> 32095
        & <extra_id_5> 32094
        NER <extra_id_6> 32093
        FET <extra_id_7> 32092
        '''
        prefix = [32099]
        if typing:
            prefix.append(32092)
            prefix.append(32097)
            for span in instances:
                spanid = [32094] + self.tokenizer.encode(span,add_special_tokens=False)
                prefix = prefix + spanid
        else:
            prefix.append(32093)
            prefix.append(32097)
            for label in instances:
                labelid = [32094] + self.tokenizer.encode(labelmap[label[0]],add_special_tokens=False) + [32096]
                labeltoken = [[self.tokenizer.encode(finelabel,add_special_tokens=False),index] for index,finelabel in enumerate(label[1])]
                labeltoken = sorted(labeltoken, key=lambda k:len(k[0]))
                cum = 0
                finelabelids = []
                for finelabel in labeltoken:
                    if len(labelid) + cum + min(len(finelabelids) - 1, 0) + len(finelabel[0]) <= 15:
                        cum += len(finelabel[0])
                        finelabelids.append(finelabel)
                    else:
                        break
                finelabelids = sorted(finelabelids,key = lambda k:k[1])
                finelabelids = [finelabel[0] for finelabel in finelabelids]
                for index,finelabel in enumerate(finelabelids):
                    if index > 0:
                        labelid.append(32095)
                    labelid = labelid + finelabel
                prefix = prefix + labelid
        prefix.append(32098)
        return prefix

    def getinstance(self,data):

        text = ' '.join(data['tokens'])
        labelmap = copy.deepcopy(self.mapping)
        if self.typing:
            entities = data['entity']
            spans = [entity['text'] for entity in entities]
            prefix = self.prefix_generator(spans,True)
        else:
            desp = data['labels']
            if not self.pred:
                desp = copy.deepcopy(desp)
                num = random.choice(list(range(self.samples[0],self.samples[1] + 1)))
                desp = random.sample(desp,num)
                targets = [i[0] for i in desp]
                generated = self.entity2text(data['entity'],labelmap, targets)
                generated = ' '.join(generated)
            prefix = self.prefix_generator(desp,labelmap=labelmap)

        inputid = self.tokenizer.encode(text,max_length=511-len(prefix),truncation=True)
        inputid = prefix + inputid
        inputmask = [1] * len(inputid)

        field = {
            'input_ids':inputid,
            'attention_mask':inputmask,
        }

        if not self.pred:

            outputid = [self.tokenizer.pad_token_id] + self.tokenizer.encode(generated,max_length=511,truncation=True)
            labels = outputid[1:]
            inputmask = [1] * len(inputid)
            outputmask = [1] * (len(outputid) - 1)

            field['decoder_input_ids'] = outputid[:-1]
            field['decoder_attention_mask'] = outputmask
            field['labels'] = labels

        return field

    def __getitem__(self, index):
        return self.getinstance(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)

class Model:
    def __init__(self, model, tokenizer, dataset, targetlabels, training_args, mapping = None):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollator(tokenizer=tokenizer)
        self.targetlabels = targetlabels
        self.label2id = {'O':0}
        for label in targetlabels:
            self.label2id[label] = len(self.label2id)
        self.mapping = mapping
        if mapping is not None:
            self.entitymap = {mapping[i]:i for i in mapping}
        self.dataset = dataset
        self.training_args = training_args
        self.id2label = {}
        for label in self.label2id:
            self.id2label[len(self.id2label)] = label
    
    def text2entity(self,text):
        entitys = []
        indexs = [i for i in range(len(text)) if text.startswith('.', i)]
        start = 0
        for i in indexs:
            subtext = text[start:i].split(' ')
            if 'is' in subtext and subtext[-1] != 'is':
                indexs2 =  [i for i,a in enumerate(subtext) if a=='is']
                index = indexs2[-1]
                entity = [' '.join(subtext[:index]).strip(),' '.join(subtext[index+1:]).strip()]
                entitys.append(entity)
                start = i + 1
        return entitys
    
    def conceptgeneration(self, single = False):
        startids = []
        if single:
            dataset = []
            for instance in self.dataset:
                for entity in instance['entity']:
                    dataset.append({'tokens':instance['tokens'],'entity':[entity]})
                    start = entity['text'] + ' is'
                    startid = [self.tokenizer.pad_token_id] + self.tokenizer.encode(start,max_length=511,truncation=True,add_special_tokens=False)
                    startids.append(torch.LongTensor(np.array([startid])))
        else:
            dataset = self.dataset
        processed_dataset = SDNetdataset(dataset, self.mapping, self.tokenizer, typing=True, pred=True)

        if single:
            data_loader = DataLoader(processed_dataset, batch_size=1, collate_fn=self.data_collator)
        else:
            data_loader = DataLoader(processed_dataset, batch_size=4, collate_fn=self.data_collator)

        max_length = 512

        model = self.model
        model.eval()
        model = model.cuda()
        decoder_start_token_id = self.tokenizer.pad_token_id
        endid = self.tokenizer.eos_token_id
        index = 0
        results = []

        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader):
                if single:
                    decoder_start_token_id = startids[index].cuda()
                outputs = model.generate(inputs=batch['input_ids'].cuda(),attention_mask=batch['attention_mask'].cuda(),max_length=max_length,num_beams=1,eos_token_id=endid, decoder_start_token_id=decoder_start_token_id)
                preds = []
                newdataset = []
                for i in range(len(outputs)):
                    preds.append(outputs[i].detach().cpu().numpy())
                    newdataset.append(dataset[index])
                    index += 1

                
                for i in range(len(preds)):
                    instance = newdataset[i]
                    tokenids = preds[i]
                    endindex = len(tokenids)
                    tokenids = [i if i >= 0 else endid for i in tokenids]
                    if endid in tokenids:
                        endindex = tokenids.index(endid)
                    tokenids = tokenids[:endindex]
                    text = self.tokenizer.decode(tokenids)
                    text = text.replace('<pad>','')
                    text = text.replace('</s>','')
                    text = text.strip()
                    if single and text[-1] != '.':
                        indexs = [i for i in range(len(text)) if text[i] == ',']
                        if len(indexs) > 0:
                            text = text[:indexs[-1]] + '.' + text[indexs[-1]+1:]
                        text = instance['entity'][0]['text'] + ' is ' + text
                    pred = self.conceptdecoder(text,[[self.tokenizer.decode(self.tokenizer.encode(entity['text'],add_special_tokens=False)),entity['type']] for entity in instance['entity']])
                    results.append(pred)
        return results
    
    def conceptdecoder(self, generated,spans):
        start = 0
        result = []
        for span in spans:
            text = span[0] + ' is '
            if text in generated[start:]:
                start = start + generated[start:].index(text) + len(text)
                if '.' in generated[start:]:
                    end = start + generated[start:].index('.')
                    label = generated[start:end].strip()
                    result.append([span,label])
        return result

    def decode(self, generated, oritokens):
        results = self.text2entity(generated)
        tokens = [self.tokenizer.decode(self.tokenizer.encode(token,add_special_tokens=False)).replace(' ','') for token in oritokens]
        tokenindex = []
        l = 0
        for token in tokens:
            for i in range(len(token)):
                tokenindex.append(l)
            l += 1
        tokenindex.append(l)
        tokens = ''.join(tokens) 
        start = 0
        preds = []
        index = 0
        for result in results:
            result[0] = result[0].replace(' ','')
            while True:
                if result[0] in tokens[start:] and len(result[0]) > 0:
                    index = tokens[start:].index(result[0])
                    offset = [tokenindex[start + index],tokenindex[start + index + len(result[0])]]
                    if offset[0] == offset[1]:
                        start = start + index + len(result[0])
                    else:
                        if result[1] in self.entitymap and self.entitymap[result[1]] in self.targetlabels:
                            preds.append({'text':' '.join(oritokens[offset[0]:offset[1]]),'offset':offset,'type':self.entitymap[result[1]]})
                            start = start + index + len(result[0])
                        break
                else:
                    break
        return preds


    
    def finetune(self):
        supportresult = self.conceptgeneration()
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)
        with open(self.training_args.output_dir + '/support_preds.json','w') as f:
            for j in supportresult:
                f.write(json.dumps(j)+'\n')

        supportmapping = {}
        for label in self.targetlabels:
            supportmapping[label] = []
        supportmapping = filterchild(supportresult,supportmapping)
        json.dump(supportmapping,open(self.training_args.output_dir + '/label_description.json','w'))

        labels = []
        for label in supportmapping:
            if label in self.targetlabels:
                finelabel = supportmapping[label]
                random.shuffle(finelabel)
                labels.append([label,finelabel])

        
        self.processed_dataset = SDNetdataset(self.dataset, self.mapping, self.tokenizer, selfdesb=[labels], samples=[min(len(self.targetlabels)//2,5),min(len(self.targetlabels),10)])
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.processed_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model()
        trainer.save_state()
        # del model
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self):
        supportmapping = json.load(open(self.training_args.output_dir + '/label_description.json'))
        labels = []
        for label in supportmapping:
            if label in self.targetlabels:
                finelabel = supportmapping[label]
                random.shuffle(finelabel)
                labels.append([label,finelabel])

        if len(labels) > 12:
            splitnum = math.ceil(len(labels)/8)
            parentnum = math.ceil(len(labels)/splitnum)
            newtestlabels = []
            repeatnum = 2
            for rep in range(repeatnum):
                random.shuffle(labels)
                for randomnum in range(splitnum):
                    newtestlabels.append(labels[randomnum*parentnum:(randomnum+1)*parentnum])
            labels = newtestlabels
        else:
            labels = [labels]

        self.processed_dataset = SDNetdataset(self.dataset, self.mapping, self.tokenizer, selfdesb = labels, pred=True)

        data_loader = DataLoader(self.processed_dataset, batch_size=self.training_args.per_device_eval_batch_size, collate_fn=self.data_collator)

        max_length = 512

        model = self.model
        model.eval()
        model = model.cuda()
        decoder_start_token_id = self.tokenizer.pad_token_id
        endid = self.tokenizer.eos_token_id
        index = 0
        with torch.no_grad():
            with open(self.training_args.output_dir + '/prediction.json', 'w') as f:
                for batch in tqdm.tqdm(data_loader):
                    outputs = model.generate(inputs=batch['input_ids'].cuda(),attention_mask=batch['attention_mask'].cuda(),max_length=max_length,num_beams=1,eos_token_id=endid, decoder_start_token_id=decoder_start_token_id,return_dict_in_generate=True,output_scores =True)
                    preds = []
                    newdataset = []
                    batch = outputs.scores[0].size(0)
                    for i in range(batch):
                        newdataset.append(self.processed_dataset.dataset[index])
                        index += 1
                    for i in range(len(outputs.scores)):
                        score = outputs.scores[i]
                        score, generated = torch.max(score,dim=-1)
                        score = score.cpu().numpy()
                        generated = generated.cpu().numpy()
                        generated = np.expand_dims(generated,1)
                        preds.append(generated)
                    preds = np.concatenate(preds,axis=1)

                    
                    for i in range(len(preds)):
                        instance = newdataset[i]
                        tokenids = preds[i]
                        endindex = len(tokenids)
                        tokenids = [i if i >= 0 else endid for i in tokenids]
                        if endid in tokenids:
                            endindex = tokenids.index(endid)
                        tokenids = tokenids[:endindex]
                        text = self.tokenizer.decode(tokenids)
                        pred = self.decode(text,instance['tokens'])
                        f.write(json.dumps({'index':instance['index'], 'generation': text,'gold': instance['entity'], 'pred': pred, 'targetlabels': self.targetlabels})+'\n')