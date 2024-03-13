import json
import logging
import os
from transformers import AutoModelForTokenClassification,AutoModel,TrainingArguments,Trainer,HfArgumentParser,AutoTokenizer, set_seed, AutoModel, AutoModelForSeq2SeqLM,AutoModelForCausalLM, AutoModelForMaskedLM, BertConfig
import numpy as np
import torch
import torch.nn as nn
import random 
import sys
import copy
import torch.nn.functional as F


from dataclasses import dataclass, field
from typing import Optional
import gc

@dataclass
class Arguments:

    plm: str = field(
        metadata={
            "help": "Pretrained model"
        },
    )

    plmpath: str = field(
        default="none", metadata={"help": "the path of model"}, 
    )

    modelname: str = field(
        default="tagmodel", metadata={"help": "the name of model"}, 
    )

    dataset: Optional[str] = field(
        default="CoNLL", metadata={"help": "dataset name"}
    )
    max_length: Optional[int] = field(
        default=128, metadata={"help": "max number of token id"}
    )
    shot: Optional[int] = field(
        default=5, metadata={"help": "shot number"}
    )
    force: Optional[bool] = field(
        default=False, metadata={"help": "force to rerun when testing"}
    )
    active: Optional[str] = field(
        default="none", metadata={"help": "active learning"}
    )
    maxshot: Optional[int] = field(
        default=5, metadata={"help": "max added shot number"}
    )
    randomseed: Optional[int] = field(
        default=42, metadata={"help": "seed"}
    )

                




logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

import importlib

from evalue import evaluefunc,writeresult,macroupdate

def activelearning(active, unlabeledindex, datasets, picknum, model, modelpath):
    if 'bertkm' in active:
        from activelearning.bertkm import Sampling
    else:
        if os.path.exists('activelearning/' + active + '.py'):
            Sampling = importlib.import_module('activelearning.' + active).Sampling
        else:
            raise Exception('active learning methods not found')
    sampling = Sampling(model, modelpath, active)
    instances = sampling.sampling(unlabeledindex, datasets, picknum)
    return instances


def load_model(modelname,plm, num_labels = None, gpt=False, tokenizer=None):
    if modelname == 'tagmodel':
        model = AutoModelForTokenClassification.from_pretrained(
            plm,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    elif modelname == 'structshot' or modelname == 'proto':
        model = AutoModel.from_pretrained(plm)
    elif modelname == 'container':
        from models.container import BertForTokenClassification
        config = BertConfig.from_pretrained(
            plm,
            num_labels = num_labels,
            id2label = {i: str(i) for i in range(num_labels)},
            label2id = {str(i): i for i in range(num_labels)},
            task_specific_params={"embedding_dimension": 128}
        )
        model = BertForTokenClassification.from_pretrained(
            plm,
            config=config
        )
    elif modelname == 'sdnet':
        model = AutoModelForSeq2SeqLM.from_pretrained(plm)
    return model

def main():
    parser = HfArgumentParser((
        Arguments,
        TrainingArguments
    ))

    args, training_args = parser.parse_args_into_dataclasses()

    modelname = args.modelname

    if os.path.exists('models/' + modelname + '.py'):
        MODEL = importlib.import_module('models.' + modelname).Model
    else:
        raise Exception('model not found')


    logger.info("Options:")
    logger.info(args)
    logger.info(training_args)

    
    seeds = os.listdir('data/' + args.dataset + '/' + str(args.shot) + 'shot')
    seeds = sorted(seeds)
    global_output_path = 'outputs/' +  args.dataset + '/' + str(args.shot) + 'shot'

    testpath = 'data/' + args.dataset + '/test.json'
    trainpath = 'data/' + args.dataset + '/train.json'

    if args.plmpath == 'none':
        args.plmath = args.plm

    targetlabels = []
    with open('data/' + args.dataset + '/' + str(args.shot) + 'shot' + '/0/targetlabel.txt') as f:
        for line in f:
            targetlabels.append(line.strip())

    if training_args.do_train:
        foundmodel = load_model(modelname, args.plmpath, len(targetlabels) + 1)
        if args.plm == 'NSP':
            tokenizer = AutoTokenizer.from_pretrained(args.plmpath, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.plmpath)
    else:
        macrof1 = None
        if modelname == 'structshot' or modelname == 'proto':
            foundmodel = load_model(modelname, args.plmpath, len(targetlabels) + 1)
            tokenizer = AutoTokenizer.from_pretrained(args.plmpath)

    for seed in seeds:
        
        set_seed(args.randomseed)

        if '.' in seed:
            continue
        datasetpath = 'data/' + args.dataset + '/' + str(args.shot) + 'shot' + '/' + seed
        output_path = global_output_path + '/' + args.plm + '/' + str(args.maxshot) + '_' + args.active

        if training_args.do_train:
            dataset = []
            with open(datasetpath+'/support.json') as f:
                for line in f:
                    dataset.append(json.loads(line))
        else:
            dataset = []
            with open(testpath) as f:
                for line in f:
                    dataset.append(json.loads(line))


        if args.active != 'none' and training_args.do_train:

            pickpath = global_output_path + '/activelearning/' + str(args.maxshot) + '_' + args.active + '/' + seed 

            if not os.path.exists(pickpath + '/pick.json'):
                
                os.makedirs(pickpath, exist_ok=True)

                pool = []
                with open(trainpath) as f:
                    for line in f:
                        pool.append(json.loads(line))
                unlabeledindex = []
                with open(datasetpath + '/unlabled.json') as f:
                    for line in f:
                        line = json.loads(line)
                        if len(line['tokens']) > 0:
                            unlabeledindex.append(line['index'])

                actmodelpath = pickpath

                if args.active == 'alps':

                    actmodel = AutoModelForMaskedLM.from_pretrained('pretrainmodels/bert-base-uncased')
                    actmodel.tokenizer = AutoTokenizer.from_pretrained('pretrainmodels/bert-base-uncased')

                elif args.active == 'bertkm':

                    actmodel = AutoModel.from_pretrained('pretrainmodels/bert-base-uncased')
                    actmodel.tokenizer = AutoTokenizer.from_pretrained('pretrainmodels/bert-base-uncased')
                

                elif args.active == 'supercd':
                    sir = AutoModel.from_pretrained('pretrainmodels/SIR')
                    sirtokenizer = AutoTokenizer.from_pretrained('pretrainmodels/SIR')

                    sir.tokenizer = sirtokenizer
                    sir.maxrate = args.shot // 2
                    sir.baned = json.load(open('pretrainmodels/SIR/notremovedname.json'))

                    ce = AutoModelForSeq2SeqLM.from_pretrained('pretrainmodels/CE')
                    cetokenizer = AutoTokenizer.from_pretrained('pretrainmodels/CE')
                    from models.sdnet import Model as CEModel

                    ce = CEModel(ce, cetokenizer, dataset, targetlabels, training_args)


                    actmodel = [ce,sir]
                    actmodelpath = global_output_path + '/supercd/' + seed
                    os.makedirs(actmodelpath, exist_ok=True)
                
                elif args.active == 'random':

                    set_seed(args.randomseed + int(seed))
                    actmodel = None
                
                elif args.active == 'fastvotek':

                    actmodel = AutoModel.from_pretrained('pretrainmodels/all-mpnet-base-v2')
                    actmodel.tokenizer = AutoTokenizer.from_pretrained('pretrainmodels/all-mpnet-base-v2')
                    actmodelpath = global_output_path + '/fastvotek/' + seed
                
                instances = activelearning(args.active, unlabeledindex, pool, len(targetlabels) * args.maxshot, actmodel, actmodelpath)

                with open(pickpath + '/pick.json', 'w') as f:
                    for instance in instances:
                        f.write(json.dumps(instance) + '\n')
            else:
                instances = []
                with open(pickpath + '/pick.json') as f:
                    for line in f:
                        instances.append(json.loads(line))
            
            assert len(instances) == args.maxshot * len(targetlabels)

            dataset = dataset + instances
        
        
            
        training_args.output_dir = output_path + '/' + seed

        set_seed(training_args.seed)

        if training_args.do_train:
            
            if args.modelname == 'sdnet':
                mapping = json.load(open('data/' + args.dataset + '/mapping.json'))
                model = MODEL(copy.deepcopy(foundmodel), tokenizer, dataset, targetlabels, training_args, mapping=mapping)
            else:
                model = MODEL(copy.deepcopy(foundmodel), tokenizer, dataset, targetlabels, training_args)

            logging.info(training_args.output_dir)
            
            model.finetune()

            del model
            gc.collect()
            torch.cuda.empty_cache()

        else:
            
            if not os.path.exists(training_args.output_dir + '/prediction.json') or args.force:

                if args.modelname != 'icl' and args.modelname != 'structshot' and args.modelname != 'proto':
                    foundmodel = load_model(modelname, training_args.output_dir, len(targetlabels) + 1)
                    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir) if args.plm != 'NSP' else AutoTokenizer.from_pretrained(training_args.output_dir, add_prefix_space=True)
                if args.modelname == 'container' or args.modelname == 'icl' or args.modelname == 'structshot' or args.modelname == 'proto':
                    support_dataset = []
                    with open(datasetpath+'/support.json') as f:
                        for line in f:
                            support_dataset.append(json.loads(line))
                    if args.active != 'none':
                        pickpath = global_output_path + '/activelearning/' + str(args.maxshot) + '_' + args.active + '/' + seed 
                        with open(pickpath + '/pick.json') as f:
                            for line in f:
                                support_dataset.append(json.loads(line))
                    model = MODEL(foundmodel, tokenizer, dataset, targetlabels, training_args, support_dataset=support_dataset)
                elif args.modelname == 'sdnet':
                
                    mapping = json.load(open('data/' + args.dataset + '/mapping.json'))
                    model = MODEL(foundmodel, tokenizer, dataset, targetlabels, training_args, mapping=mapping)
                else:
                    model = MODEL(foundmodel, tokenizer, dataset, targetlabels, training_args)
                model.predict()
                del model
                gc.collect()
                torch.cuda.empty_cache()

            results = []
            with open(training_args.output_dir + '/prediction.json') as f:
                for line in f:
                    line = json.loads(line)
                    results.append(line)

            f1 = evaluefunc(results,targetlabels)
            writeresult(training_args.output_dir,f1)
            macrof1 = macroupdate(f1,macrof1)

    if not training_args.do_train:
        macroresult = {
            'p':np.mean(macrof1['p']),
            'r':np.mean(macrof1['r']),
            'f1':np.mean(macrof1['f1']),
            'std':np.std(macrof1['f1'])
        }
        if 'typef1' in macrof1:
            macroresult['typef1'] = {}
            for label in macrof1['typef1']:
                macroresult['typef1'][label] = {}
                macroresult['typef1'][label]['p'] = np.mean(macrof1['typef1'][label]['p'])
                macroresult['typef1'][label]['r'] = np.mean(macrof1['typef1'][label]['r'])
                macroresult['typef1'][label]['f1'] = np.mean(macrof1['typef1'][label]['f1'])
                macroresult['typef1'][label]['std'] = np.std(macrof1['typef1'][label]['f1'])
        macroresult['origin'] = macrof1
        writeresult(output_path,macroresult)
        sys.stdout.write('macrop:{0:.4f}, macror:{1:.4f}, macrof1: {2:.4f}, std: {3:.4f}'.format(macroresult['p'],macroresult['r'],macroresult['f1'],macroresult['std']) + '\r')
        sys.stdout.write('\n')
        print('\n')
                


if __name__ == "__main__":
    main()