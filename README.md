# shadow-free-MIAs
This respository contains the source code of the paper "Shadow-Free Membership Inference Attacks: Recommender Systems Are More Vulnerable Than You Thought" which is submitted to the IJCAI 2024. Authors proposed a novel membership inference attack against recommender systems without shadow training. 
# Requirement
* torch == 2.1.1
* python == 3.9.18
* numpy == 1.26.0
* pandas == 2.1.3


# Dataset
The experiments are evaluated on three benchmark datasets, i.e., MovieLens-1M, Amazon Beauty, and Ta-feng. 
* For MovieLens-1M, download dataset from [here](https://grouplens.org/datasets/movielens/1m/), then put it into the path "/dataprocess/"
* For Amazon Beauty, download dataset from [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews), then put it into the path "/dataprocess/"
* For Ta-feng, download dataset from [here](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset), then put it into the path "/dataprocess/"

# Recommender System
* Traditional recommender system
  * For Item-based Collaborative Filtering(ICF), follow this [link](https://librecommender.readthedocs.io/en/latest/tutorial.html) to implement ICF.
* Advanced deep learning based recommender systems
  * NCF, follow this [link](https://librecommender.readthedocs.io/en/latest/tutorial.html) to implement NCF.
  * BERT4Rec, follow this [link](https://github.com/WZH-NLP/DL-MIA-KDD-2022/tree/main/DL-MIA-SR/Recommender/BERT4Rec-Pytorch-master) to implement BERT4Rec.
  * Caser, follow this [link](https://github.com/WZH-NLP/DL-MIA-KDD-2022/tree/main/DL-MIA-SR/Recommender/caser_pytorch-master) to implement Caser.
  * GRU4Rec, follow this [link](https://github.com/WZH-NLP/DL-MIA-KDD-2022/tree/main/DL-MIA-SR/Recommender/GRU4REC-pytorch-master) to implement GRU4Rec.


# Get started
The following command can be used to train shadow-free MIAs for both traditional recommender systems and advanced deep learning based recommender systems:
```
cd attack/SFMD/attackModel/
python beauty_Bert4Rec.py
```
