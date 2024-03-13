import json, tqdm, sys, copy

def getmetric(prednum,goldnum,tt,classprednum=None,classgoldnum=None,classtt=None):
    p = 0
    r = 0
    f1 = 0
    if prednum > 0:
        p = tt/prednum
    if goldnum > 0:
        r = tt/goldnum
    if p > 0 and r > 0:
        f1 = 2*p * r / (p+r)
    result = {'p':p,'r':r,'f1':f1}
    if classprednum is not None:
        result['typef1'] = {}
        for label in classprednum:
            p = 0
            r = 0
            f1 = 0
            if classprednum[label] > 0:
                p = classtt[label]/classprednum[label]
            if classgoldnum[label] > 0:
                r = classtt[label]/classgoldnum[label]
            if p > 0 and r > 0:
                f1 = 2*p * r / (p+r)
            result['typef1'][label] = {'p':p,'r':r,'f1':f1}
    return result

def macroupdate(f1,macrof1=None):
    if macrof1 is None:
        macrof1 = {
            'p': [f1['p']],
            'r': [f1['r']],
            'f1': [f1['f1']],
        }
        if 'typef1' in f1:
            macrof1['typef1'] = {}
            for label in f1['typef1']:
                macrof1['typef1'][label] = {}
                macrof1['typef1'][label]['p'] = [f1['typef1'][label]['p']]
                macrof1['typef1'][label]['r'] = [f1['typef1'][label]['r']]
                macrof1['typef1'][label]['f1'] = [f1['typef1'][label]['f1']]
    else:
        macrof1['p'].append(f1['p'])
        macrof1['r'].append(f1['r'])
        macrof1['f1'].append(f1['f1'])
        if 'typef1' in macrof1:
            for label in f1['typef1']:
                macrof1['typef1'][label]['p'].append(f1['typef1'][label]['p'])
                macrof1['typef1'][label]['r'].append(f1['typef1'][label]['r'])
                macrof1['typef1'][label]['f1'].append(f1['typef1'][label]['f1'])

    return macrof1

def filterpred(preds):
    textoffset = []
    if len(preds) == 1:
        return preds[0]
    offsetnum = {}
    for pred in preds:
        offset = []
        for entity in pred:
            entitytype = entity['type']
            if entitytype not in offsetnum:
                offsetnum[entitytype] = {}
            for i in range(entity['offset'][0],entity['offset'][1]):
                offset.append(i)
                if i not in offsetnum[entitytype]:
                    offsetnum[entitytype][i] = 1
                else:
                    offsetnum[entitytype][i] += 1
    newpred = []
    for pred in preds:
        for entity in pred:
            entitytype = entity['type']
            maxthisnum = 0
            maxothernum = 0
            for i in range(entity['offset'][0],entity['offset'][1]):
                if i in textoffset:
                    maxothernum = 3
                    break
                maxthisnum = max(maxthisnum,offsetnum[entitytype][i])
                for j in offsetnum:
                    if j != entitytype and i in offsetnum[j]:
                        maxothernum = max(maxothernum,offsetnum[j][i])
            if maxthisnum > maxothernum:
                newpred.append(entity)
                for i in range(entity['offset'][0],entity['offset'][1]):
                    textoffset.append(i)
    return newpred

def writeresult(modelfile,f1):
    filename = ['/f1','/error','/errorinfo','/wrongmap']
    json.dump(f1,open(modelfile+filename[0]+'.json','w'))

def evaluefunc(results,targetlabel):
    print('evalue')
    f1 = []

    lastpred = []
    lastdata = 0
    lastindex = 0
    prednum = 0
    goldnum = 0
    tt = 0


    classprednum = {}
    classgoldnum = {}
    classtt = {}
    wrongmap = {}
    for label in targetlabel:
        classprednum[label] = 0
        classgoldnum[label] = 0
        classtt[label] = 0
        wrongmap[label] = {}
    
    results.append({'index':-1,'pred':[]})
    
    for result in tqdm.tqdm(results):
        index = result['index']
        if index == lastindex:
            lastdata = result
            pred = result['pred']
            lastpred.append(pred)
            continue
        else:
            data = lastdata
            gold = data['gold']


            lastpred = filterpred(lastpred)
            prednum += len(lastpred)
            goldnum += len(gold)


            for entity in gold:
                classgoldnum[entity['type']] += 1
            for entity in lastpred:
                classprednum[entity['type']] += 1
            
            predentitytext = [[j['text'],j['offset']] for j in lastpred]

            for entityindex,entity in enumerate(gold):
                if entity in lastpred:
                    tt += 1
                    classtt[entity['type']] += 1
                else:
                    if [entity['text'],entity['offset']] in predentitytext:
                        newentity = copy.deepcopy(entity)

            pred = result['pred']
            lastpred = [pred]
            lastindex = index
            lastdata = result
    f1 = getmetric(prednum,goldnum,tt,classprednum,classgoldnum,classtt)
    sys.stdout.write('p:{0:.4f}, r:{1:.4f}, f1: {2:.4f}'.format(
    f1['p'],f1['r'], f1['f1']) + '\r')
    sys.stdout.write('\n')
    print('\n')
    return f1
