import os
import logging
import json
from transformers import DataCollatorForTokenClassification,Trainer
import numpy as np
import torch,tqdm
import gc
logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model, tokenizer, dataset, targetlabels, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.targetlabels = targetlabels
        self.label2id = {'O':0}
        for label in targetlabels:
            self.label2id[label] = len(self.label2id)
        self.processed_dataset = self.preprocessdata(dataset)
        self.dataset = dataset
        self.training_args = training_args
        self.id2label = {}
        for label in self.label2id:
            self.id2label[len(self.id2label)] = label
    
    def preprocessdata(self, dataset):
        processed_dataset = {i:[] for i in dataset[0]}
        for instance in tqdm.tqdm(dataset):
            for key in instance:
                processed_dataset[key].append(instance[key])
        processed_dataset = self.tokenize_and_align_labels(processed_dataset)
        processed_dataset2 = []
        for i in range(len(processed_dataset['input_ids'])):
            processed_dataset2.append({key:processed_dataset[key][i] for key in processed_dataset})
        return processed_dataset2

    def align_labels_with_tokens(self, labels, word_ids):
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

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], max_length=128, truncation=True, is_split_into_words=True
        )

        new_labels = []
        for i,entities in enumerate(examples['entity']):
            labels = [0] * len(examples['tokens'][i])
            for entity in entities:
                for index in range(entity['offset'][0],entity['offset'][-1]):
                    labels[index] = self.label2id[entity['type']]
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

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
        trainer = Trainer(
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

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        result = trainer.predict(self.processed_dataset)
        predictions = result.predictions
        predictions = np.argmax(predictions,axis=-1)
        labels = result.label_ids

        pred_entities = []
        index = 0
        for label, predict in zip(labels,predictions):
            pred_entities.append(self.tag2entity(self.id2label,predict,label,self.dataset[index]['tokens']))
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
