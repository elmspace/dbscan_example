"""
	Author: Ash D
	Date: Sept/2019
	Description: This is a simple example demonstrating the
				 unsupervised clustering algorithm DBSCAN.
				 For more infomration see www.elmspace.com
"""

import random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def cluster_finder(points, eps, min_samples):
	"""
	This function will use sklearn DBSCAN and Matplotlib to
	find clusters in points and plot them.
	Input:
		:points: List of x,y coordinates <list of lists>
		:eps: Minimum distance used by DBSCAN for clusteting <float>
		:min_samples: Minimum number points required per cluster (used by DBSCAN algo) <int>
	"""
	X = np.array(points)
	clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
	labels = clustering.labels_
	labels = [i+1 for i in labels]
	colors = cm.rainbow(np.linspace(0, 1, len(set(labels))+1))
	plt.scatter(X[:,0], X[:,1], c=colors[labels])
	plt.show()

def create_clusters(nc, npc, bs, delta):
	"""
	This function creates a list of [x,y] points based on:
	Input:
		:nc: Number of Clusters <int>
		:npc: Number of points per cluster, <int>
		:bs: Box size <int>
		:delta: Size around the center of each cluster point to distribute the
				cluster points <int>
	Output:
		:points: List of x,y coordinates. <list of lists>
	"""
	points = []
	for i in range(nc):
		x = random.randint(0,bs)
		y = random.randint(0,bs)
		for j in range(npc):
			delta_x = random.uniform(-delta, delta)
			delta_y = random.uniform(-delta, delta)
			points.append([x+delta_x, y+delta_y])
	for i in range(nc * 100):
		points.append([random.randint(0,bs), random.randint(0,bs)])
	return points

if __name__ == "__main__":
	nc = 8 # Number of clustersc
	npc = 50 # Number of points per cluster
	bs = 100 # Box size
	delta = 2 # Distribution around cluster centers
	points = create_clusters(nc, npc, bs, delta)
	cluster_finder(points, 3, 20)