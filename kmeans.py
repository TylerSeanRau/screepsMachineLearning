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

kmeans = KMeans(n_clusters=1)
kmeans.fit(dfWithoutLabel)
print("Center")
print(kmeans.cluster_centers_[0])

print("Average dsitance to center")
dist = DistanceMetric.get_metric('euclidean')
distancesToCenter=np.concatenate(dist.pairwise(dfWithoutLabel.values,kmeans.cluster_centers_))
print(np.average(distancesToCenter))

print("Most anomalous lines")
sortedDistancesToCenter = np.argsort(distancesToCenter)[::-1]
for i in range(0,5):
    print(sortedDistancesToCenter[i])
    print(dfWithoutLabel.values[sortedDistancesToCenter[i]])

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
                color='r', zorder=10)
#Mark labeled points
for i in range(0,len(dfWithoutLabel)):
    if(df.values[i][df.columns.get_loc('anomaly')] == True):
        plt.scatter(dfWithoutLabel.values[i][dfWithoutLabel.columns.get_loc(xToGraph)],dfWithoutLabel.values[i][dfWithoutLabel.columns.get_loc(yToGraph)], marker='x', s=169, linewidths=3,color='g', zorder=10)
plt.show()
