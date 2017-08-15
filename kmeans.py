import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric

parser = argparse.ArgumentParser()
parser.add_argument("room", help="room name")
args = parser.parse_args()

df = pd.read_csv("./screepsData/" + args.room)
kmeans = KMeans(n_clusters=1)
kmeans.fit(df)
print("Center")
print(kmeans.cluster_centers_[0])
dist = DistanceMetric.get_metric('euclidean')
distancesToCenter=np.concatenate(dist.pairwise(df.values,kmeans.cluster_centers_))
print("Average dsitance to center")
print(np.average(distancesToCenter))

print("Most anomalous lines")
sortedDistancesToCenter = np.argsort(distancesToCenter)[::-1]
for i in range(0,5):
    print(sortedDistancesToCenter[i])
