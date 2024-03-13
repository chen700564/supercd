import random, copy

class Sampling:
    def __init__(self,model, modelpath, way):
        self.model = model
        self.modelpath = modelpath
        self.way = way

    def sampling(self, unlabeledindex, datasets, picknum):
        indexs = copy.deepcopy(unlabeledindex)
        random.shuffle(indexs)

        addedtokens = []
        result = []
        for index in indexs:
            if index not in unlabeledindex:
                continue
            result.append(datasets[index])
            if len(result) == picknum:
                break
        return result