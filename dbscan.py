import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric

parser = argparse.ArgumentParser()
parser.add_argument("room", help="room name")
args = parser.parse_args()

df = pd.read_csv("./screepsData/" + args.room)

db = DBSCAN(eps=500, min_samples=10)
db.fit(df)

#print(db.labels_)
for i in range(0,len(df)):
    if db.labels_[i] == -1:
        print(i)
