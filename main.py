from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from collections import Counter

def cluster_finder(points):
	X = np.array(points)

	clustering = DBSCAN(eps=3, min_samples=20).fit(X)
	labels = clustering.labels_
	labels = [i+1 for i in labels]

	cluster_labels = Counter(labels)
	colors = cm.rainbow(np.linspace(0, 1, len(set(labels))+1))

	fig = plt.figure(figsize=(10,10))
	plt.scatter(X[:,0], X[:,1], c=colors[labels])
	fig.savefig('temp.png', dpi=fig.dpi)


def create_clusters(nc, npc, bs, delta):
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

	nc = 5
	npc = 50
	bs = 100
	delta = 2
	points = create_clusters(nc, npc, bs, delta)
	cluster_finder(points)