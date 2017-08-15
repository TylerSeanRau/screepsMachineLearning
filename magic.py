import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()

df = pd.read_csv("./screepsData/W79N98")
kmeans = KMeans()
kmeans.fit(df)
