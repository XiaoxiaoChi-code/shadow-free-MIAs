import random
import Evaluation
import LFM

def readData():
    data = []
    # reading dataset ml-1m
    fileName = '../processed_movielens/ml-1m_vector'

    # reading dataset amazon beauty
    # fileName = '../processed_amazon/beauty_vector'

    # reading dataset ta-feng
    # fileName = "../processed_ta-feng/ta-feng_vector"

    fr = open(fileName, 'r')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        if lineArr[0] != 'SessionID':
            data.append([lineArr[0], lineArr[1], 1.0])
    return data

def SplitData(data, M, k, seed):
    # test = []
    train = []
    random.seed(seed)
    for user, item, rating in data:
        train.append([user, item, rating])
        # if random.randint(0, M - 1) == k:
        #     test.append([user, item, rating])
        # else:
        #     train.append([user, item, rating])
    return train

def SplitData_test(data, M, k, seed):
    test = []
    # train = []
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M - 1) == k:
            test.append([user, item, rating])
        # else:
        #     train.append([user, item, rating])
    return test

def transform(oriData):
    ret = dict()
    for user, item, rating in oriData:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating
    return ret




if __name__ == "__main__":
    data = readData()
    numFlod = 1
    precision = 0
    recall = 0
    coverage = 0
    popularity = 0
    for i in range(0, numFlod):
        oriTrain = SplitData(data, numFlod, i, 0)
        oriTest = SplitData_test(data, numFlod, i, 0)
        train = transform(oriTrain)
        test = transform(oriTest)

        # change the length of latent vector. (the second parameter)
        [P, Q] = LFM.LatentFactorModel(train, 90, 30, 0.02, 0.01)

        # for dataset ml-1m
        fw = open('../Vectorized_itemEmbed/ml-1m_itemMatrix_pre_90', 'w')
        for key, value in Q.items():
            fw.write(str(key)+'\t'+str(value)+'\n')

        # for dataset amazon beauty
        # fw = open('../Vectorized_itemEmbed/beauty_itemMatrix_pre_90', 'w')
        # for key, value in Q.items():
        #     fw.write(str(key) + '\t' + str(value) + '\n')



        # for dataset ta-feng
        # fw = open('../Vectorized_itemEmbed/ta-feng_itemMatrix_pre_90', 'w')
        # for key, value in Q.items():
        #     fw.write(str(key) + '\t' + str(value) + '\n')

        rank = LFM.Recommend('2', train, P, Q)
        result = LFM.Recommendation(test.keys(), train, P, Q)

        N = 10
        precision += Evaluation.Precision(train, test, result, N)
        recall += Evaluation.Recall(train, test, result, N)
        coverage += Evaluation.Coverage(train, test, result, N)
        popularity += Evaluation.Popularity(train, test, result, N)

    precision /= numFlod
    recall /= numFlod
    coverage /= numFlod
    popularity /= numFlod

    
    print('precision = %f' % precision)
    print('recall = %f' % recall)
    print('coverage = %f' % coverage)
    print('popularity = %f' % popularity)
