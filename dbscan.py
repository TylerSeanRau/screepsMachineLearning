import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric

parser = argparse.ArgumentParser()
parser.add_argument("room", help="room name")
parser.add_argument("e", help="epsilon", type=int)
parser.add_argument("minpoints", help="minpoints", type = int)
args = parser.parse_args()

df = pd.read_csv("./screepsData/" + args.room)
dfWithoutLabel = df.drop(['anomaly'],axis=1) if 'anomaly' in df.columns else df

weights = [['num_creeps',5000]]
for i in range(0,len(weights)):
    for j in range(0,len(dfWithoutLabel)):
        dfWithoutLabel.values[j][dfWithoutLabel.columns.get_loc(weights[i][0])] *= weights[i][1]

db = DBSCAN(eps=args.e, min_samples=args.minpoints)
db.fit(dfWithoutLabel)

# print(dfWithoutLabel.values[0])
# print("test")
# distToNext = array("i")
# dist = DistanceMetric.get_metric('euclidean')
# for i in range(0,len(df)-1):
#     print()
# distancesToCenter=np.concatenate(dist.pairwise(df.values,kmeans.cluster_centers_))
# print("Average dsitance to center")
# print(np.average(distancesToCenter))
#print(db.labels_)



if 'anomaly' in df.columns:
    correct = 0
    ta = 0
    ca = 0
    for i in range(0,len(df)):
        if df.values[i][-1] == True:
            ta += 1
            if db.labels_[i] == -1:
                correct+=1
                ca+=1
        else:
            if db.labels_[i] != -1:
                correct+=1
    print("%.2f %.2f" % (ca/ta,correct/len(df)),end='')
else:
    for i in range(0,len(df)):
        if db.labels_[i] == -1:
            print(i)
