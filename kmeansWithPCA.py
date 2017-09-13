import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA, KernelPCA

parser = argparse.ArgumentParser()
parser.add_argument("room", help="room name")
args = parser.parse_args()

df = pd.read_csv("./screepsData/" + args.room)
dfWithoutLabel = df.drop(['anomaly'],axis=1) if 'anomaly' in df.columns else df

pca = PCA(n_components=2)
newdata = pca.fit_transform(dfWithoutLabel)

kmeans = KMeans(n_clusters=1)
kmeans.fit(newdata)
print("Center")
print(kmeans.cluster_centers_[0])
print("Average dsitance to center")
dist = DistanceMetric.get_metric('euclidean')
distancesToCenter=np.concatenate(dist.pairwise(newdata,kmeans.cluster_centers_))
print(np.average(distancesToCenter))

sortedDistancesToCenter = np.argsort(distancesToCenter)[::-1]
#Mark 5 furthest points
for i in range(0,5):
    plt.scatter(newdata[sortedDistancesToCenter[i]][0],newdata[sortedDistancesToCenter[i]][1],
                    marker='x', s=169, linewidths=3,
                    color='r', zorder=10)
#Add all points, use X + dot if the point was labeled anomaly otherwise just dot
for i in range(0,len(newdata)):
    if(df.values[i][df.columns.get_loc('anomaly')] == True):
        plt.scatter(newdata[i][0],newdata[i][1],marker='x', s=169, linewidths=3, color='g', zorder=10)
        plt.scatter(newdata[i][0],newdata[i][1], color="b")
    else:
        plt.scatter(newdata[i][0],newdata[i][1], color="b")