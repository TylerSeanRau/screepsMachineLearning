import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric

parser = argparse.ArgumentParser()
parser.add_argument("room", help="room name")
args = parser.parse_args()

df = pd.read_csv("./screepsData/" + args.room)
dfWithoutLabel = df.drop(['anomaly'],axis=1) if 'anomaly' in df.columns else df

weights = [['num_creeps',5000]]
for i in range(0,len(weights)):
    for j in range(0,len(dfWithoutLabel)):
        dfWithoutLabel.values[j][dfWithoutLabel.columns.get_loc(weights[i][0])] *= weights[i][1]

kmeans = KMeans(n_clusters=1)
kmeans.fit(dfWithoutLabel)
#print("Center")
#print(kmeans.cluster_centers_[0])
dist = DistanceMetric.get_metric('euclidean')
distancesToCenter=np.concatenate(dist.pairwise(dfWithoutLabel.values,kmeans.cluster_centers_))
print("Average dsitance to center")
print(np.average(distancesToCenter))

print("Most anomalous lines")
sortedDistancesToCenter = np.argsort(distancesToCenter)[::-1]
for i in range(0,5):
    print(sortedDistancesToCenter[i])

xToGraph = 'num_creeps'
yToGraph = 'creep_energy'
plt.scatter(dfWithoutLabel[[xToGraph]],dfWithoutLabel[[yToGraph]])
plt.xlabel(xToGraph)
plt.ylabel(yToGraph)
#Mark Center of the graph
plt.scatter(kmeans.cluster_centers_[0][dfWithoutLabel.columns.get_loc(xToGraph)],kmeans.cluster_centers_[0][dfWithoutLabel.columns.get_loc(yToGraph)],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
#Mark Most anomalous points
for i in range(0,5):
    plt.scatter(dfWithoutLabel.values[sortedDistancesToCenter[i]][dfWithoutLabel.columns.get_loc(xToGraph)],dfWithoutLabel.values[sortedDistancesToCenter[i]][dfWithoutLabel.columns.get_loc(yToGraph)],
                marker='x', s=169, linewidths=3,
                color='g', zorder=10)
plt.show()