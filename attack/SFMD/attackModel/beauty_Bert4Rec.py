import torch
import pandas as pd
import time
import numpy as np

# using Beauty target dataset, recommender model Bert4Rec


# pathS = "../../../recommender/BERT4Rec-Pytorch-master/mydata/processed/"
Vpath = "../../../dataprocess/Vectorized_itemEmbed"
path = "../../../dataprocess/processed_amazon/"

pathS = ""

num_latent = 100

itemDict = {}
with open(path + "beauty_itemDict", 'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        itemDict[int(line[1])] = int(line[0])

fr_vector_shadow = open(Vpath + "/beauty_itemMatrix.txt", 'r')

vectors1 = {}  # vectors for shadow items
index = -1
for line in fr_vector_shadow.readlines():
    index = index + 1
    line = line.split(' ')
    line = list(map(float, line))
    t_vectors = torch.tensor(line)
    vectors1[itemDict[index]] = t_vectors

print("finished1")



filePath = pathS + "beauty_Tmember_train"
data = pd.read_csv(filePath, sep=',', header = None, low_memory=False, skiprows=1)
# print(data.head())
data.columns = ['SessionID', 'ItemID', 'Rating', 'Time']
# print(data.head())
data = data.astype(int)
Popularity = data.groupby('ItemID').size()
sorted_Popularity = Popularity.sort_values(ascending=False)
popular_item = sorted_Popularity.index.tolist()
# print(popular_item)
# print(popular_item)
temp_vector = torch.zeros(num_latent)
count = i = 0
while count < 100:
    if popular_item[i] in vectors1.keys():
        temp_vector = temp_vector + vectors1[popular_item[i]]
        count = count + 1
        i = i + 1
    else:
        i = i + 1

vector_shadow_baseline_recommend = temp_vector/count


print("finished2")



Smember_rec = open(pathS + "beauty_Tmember_recommendations", 'r')
recommend_Smember = {}
for line in Smember_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Smember.setdefault(int(sessionID), []).append(int(itemID))
# print("length of recommend_Smember.keys:", len(recommend_Smember.keys()))
# print("recommend_Smember.keys:", sorted(recommend_Smember.keys()))


# read recommendations for shadow non-members
Snonmem_rec = open(pathS + "beauty_Tnonmem_recommendation", 'r')
recommend_Snonmem = {}
for line in Snonmem_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Snonmem.setdefault(int(sessionID), []).append(int(itemID))



# read interactions for shadow members and shadow non-members
itm = open(pathS + "beauty_Tmember_train", 'r')
itn = open(pathS + "beauty_Tnonmem_train", 'r')


interaction_Smember = {}  # interactions for shadow_member
interaction_Snonmem = {}  # interactions for shadow_nonmem

for line in itm.readlines():
    line = line.split('\t')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Smember.setdefault(int(sessionID), []).append(int(itemID))
# print("length of interactions_Smember.keys:", len(interaction_Smember.keys()))
# print("interactions_Smember.keys:", sorted(interaction_Smember.keys()))

for line in itn.readlines():
    line = line.split(',')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Snonmem.setdefault(int(sessionID), []).append(int(itemID))




time1 = time.time()
# getting shadow members' feature vectors
label_shadow_member = {}
memSimilarity1 = {}
memSimilarity2 = {}
member_S1 = []
member_S2 = []
memberS1minusS2 = []

member_count = 0
num_of_member = 0
vector_shadow_member1 = {}  # vectors for shadow member interaction
for key, value in interaction_Smember.items():
    # key是userID， value为推荐列表
    label_shadow_member[key] = torch.tensor([1])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_member1[key] = temp_vector

vector_shadow_member2 = {}  # vectors for shadow member recommendation
for key, value in recommend_Smember.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_member2[key] = temp_vector

    memSimilarity1[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_shadow_member1[key], vector_shadow_member2[key]), 2), dim=0)).tolist())
    member_S1.append(memSimilarity1[key])

    memSimilarity2[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_shadow_member2[key], vector_shadow_baseline_recommend), 2), dim=0)).tolist())
    member_S2.append(memSimilarity2[key])

    memberS1minusS2.append(memSimilarity1[key] - memSimilarity2[key])

    num_of_member = num_of_member + 1
    if memSimilarity1[key] > memSimilarity2[key]:
        member_count = member_count + 1

np.savetxt("../MetricAttackResults/AB_memberS1.csv", member_S1, delimiter=',')
np.savetxt("../MetricAttackResults/AB_memberS2.csv", member_S2, delimiter=',')
np.savetxt("../MetricAttackResults/AB_memberS1minusS2.csv", memberS1minusS2, delimiter=',')


# getting shadow non-members' feature vectors
label_shadow_nonmem = {}
nonmemSimilarity1 = {}
nonmemSimilarity2 = {}

nonmember_S1 = []
nonmember_S2 = []
nonmemS1minusS2 = []

nonmem_count = 0
num_of_nonmem = 0

vector_shadow_nonmem1 = {}  # vectors for shadow nonmember interaction
for key, value in interaction_Snonmem.items():
    # key是userID， value为交互历史
    label_shadow_nonmem[key] = torch.tensor([0])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_nonmem1[key] = temp_vector

vector_shadow_nonmem2 = {}  # vectors for shadow nonmember recommendation
for key, value in recommend_Snonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_nonmem2[key] = temp_vector

    nonmemSimilarity1[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_shadow_nonmem1[key], vector_shadow_nonmem2[key]), 2), dim=0)).tolist())
    nonmember_S1.append(nonmemSimilarity1[key])

    nonmemSimilarity2[key] = 1.0 / (torch.sqrt(torch.sum(torch.pow(torch.subtract(vector_shadow_nonmem2[key], vector_shadow_baseline_recommend), 2), dim=0)).tolist() + 0.0001)
    nonmember_S2.append(nonmemSimilarity2[key])

    nonmemS1minusS2.append(nonmemSimilarity1[key] - nonmemSimilarity2[key])

    num_of_nonmem = num_of_nonmem + 1
    if nonmemSimilarity1[key] < nonmemSimilarity2[key]:
        nonmem_count = nonmem_count + 1

np.savetxt("../MetricAttackResults/AB_nonmemS1.csv", nonmember_S1, delimiter=",")
np.savetxt("../MetricAttackResults/AB_nonmemS2.csv", nonmember_S2, delimiter=",")
np.savetxt("../MetricAttackResults/AB_nonmemS1minusS2.csv", nonmemS1minusS2, delimiter=",")

time = time.time() - time1
print("Time cost is {}".format(time))


ASR = (member_count + nonmem_count)/(num_of_member + num_of_nonmem)

falsePositive = num_of_nonmem - nonmem_count
falsePositiveRate = falsePositive / num_of_nonmem

truePositiveRate = member_count / num_of_member

print("the number of members when S1 > S2 is {}".format(member_count))
print("total number of members is {}".format(num_of_member))
print("the number of non-members when S1 < S2 is {}".format(nonmem_count))
print("the total number of nonmembers is {}".format(num_of_nonmem))
print("Attack success rate is {}".format(ASR))
print("false positive rate is {}".format(falsePositiveRate))

result_file = open("../MetricAttackResults/beauty_Bert4Rec.txt", 'w')
result_file.write("the number of members when S1 > S2 is " + "\t" + str(member_count) + "\n")
result_file.write("total number of members is " + "\t" + str(num_of_member) + "\n")
result_file.write("the number of non-members when S1 < S2 is " + "\t" + str(nonmem_count) + "\n")
result_file.write("the total number of nonmembers is " + "\t" + str(num_of_nonmem) + "\n")
result_file.write("Attack success rate is " + "\t" + str(ASR) + "\n")
result_file.write("false positive rate is " + "\t" + str(falsePositiveRate) + "\n")

result_file.write("true positive rate is " + "\t" + str(truePositiveRate) + "\n")

result_file.write("Time cost is " + "\t" + str(time) + "\n")

result_file.close()
