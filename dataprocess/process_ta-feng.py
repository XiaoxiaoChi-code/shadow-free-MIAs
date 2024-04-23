import numpy as np
import pandas as pd
import random
import torch


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def removeShortSeq(data):
    """
    parameter data: dataframe type
    return filtered dataframe type data
    """
    seqLen = data.groupby('UserID').size()
    # according to the KDD paper, the minimum interaction for ech user is 20
    data = data[np.in1d(data['UserID'], seqLen[seqLen > 4].index)]
    return data



def step0(filename, filename1):
    data = pd.read_csv(filename, sep=",", header=None, skiprows=1, engine="python")
    data.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    data = removeShortSeq(data)
    itemLen = data.groupby('ItemID').size()
    data = data[np.in1d(data['ItemID'], itemLen[itemLen>4].index)]
    data = removeShortSeq(data)
    data.to_csv(filename1, sep=",", index=False, header=None)
    print("users: ", data['UserID'].nunique())
    print("Items: ", data['ItemID'].nunique())


def step1(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    data.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    userGroup = data.groupby('UserID')
    path = "processed_ta-feng/"
    num_target = num_shadow = num_vector = 0

    for userID, userSeq in userGroup:
        rand = random.randint(1,3)
        if rand == 1:
            userSeq.to_csv(path+'ta-feng_vector_pre', sep=',', index=False, mode='a', header=None)
            num_vector = num_vector + 1
        elif rand == 2:
            userSeq.to_csv(path+'ta-feng_shadow_pre', sep=',', index=False, mode='a', header=None)
            num_shadow = num_shadow + 1
        else:
            userSeq.to_csv(path+'ta-feng_target_pre', sep=',', index=False, mode='a', header=None)
            num_target = num_target + 1
    print("num_vector: ", num_vector)
    print("num_shadow: ", num_shadow)
    print("num_target: ", num_target)



# step2: re-index for users
def step2(fr, fw):
    path = "processed_ta-feng/"
    f1 = pd.read_csv(path+fr, sep=',')
    f1.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']
    f2 = open(path+fw, "w")
    userGroup = f1.groupby('UserID')
    user_index = -1
    f2.write('SessionID'+','+'ItemID'+','+'Rating'+','+'Time'+'\n')
    for user, Items in userGroup:
        user_index = user_index + 1
        ItemIds = Items['ItemID'].to_list()
        Ratings = Items['Rating'].to_list()
        TimeStamps = Items['TimeStamp'].to_list()
        for i in range(len(ItemIds)):
            f2.write(str(user_index)+','+str(ItemIds[i]) + ',' + str(Ratings[i]) + ',' + str(TimeStamps[i]) + '\n')
    f2.close()
    return user_index


def step3(fr, fw1, fw2):
    path = "processed_ta-feng/"
    f2 = open(path+fw1, 'w')
    f3 = open(path+fw2, 'w')

    ItemDict = {}
    ItemSet = set()

    item_index = 0
    first_line = True

    with open(path+fr, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            SessionID = line[0]
            ItemID = line[1]
            Rating = line[2]
            if first_line:
                f2.write(str(SessionID) + '\t' +str(ItemID) +'\t' + str(Rating) +'\n')
                first_line = False
            else:
                if ItemID not in ItemSet:
                    ItemSet.add(ItemID)
                    ItemDict[ItemID] = item_index
                    item_index = item_index + 1
                f2.write(str(SessionID) +'\t' + str(ItemDict[ItemID]) + '\t' + str(Rating) + '\n')
    for key, value in ItemDict.items():
        f3.write(str(key) + '\t' + str(value) + '\n')
    f2.close()
    f3.close()

    return item_index







if __name__ == "__main__":



    set_seed(2021)
    # drop unnecessary columns in original dataset
    filename = 'processed_ta-feng_data.csv'
    filename1 = "processed_ta-feng/ratings_after_step0.dat"
    step0(filename, filename1)
    step1(filename1)

    fr_target = "ta-feng_target_pre"
    fr_shadow = "ta-feng_shadow_pre"
    fr_vector = "ta-feng_vector_pre"

    fw_target = "ta-feng_target"
    fw_shadow = "ta-feng_shadow"
    fw_vector = "ta-feng_vector_pre1"

    target_num = step2(fr_target, fw_target)
    print("target dataset is finished")
    shadow_num = step2(fr_shadow, fw_shadow)
    print("shadow dataset is finished")
    vector_num = step2(fr_vector, fw_vector)
    print("dataset for items is finished")
    print("target_num: {}, shadow_num: {}, vector_num: {}".format(target_num, shadow_num, vector_num))

    fw_vector1 = "ta-feng_vector"
    fw_itemdict = "ta-feng-itemDict"
    item_num_of_vectorize = step3(fw_vector, fw_vector1, fw_itemdict)
    print("item_num_of_vectorize:", item_num_of_vectorize)
