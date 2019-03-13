import os
import sys
from heapq import heappush, heappop
import numpy as np
from numpy import *
import pandas as pd
from scipy.spatial import distance
import heapq
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def load_data(data_path):
	# Current directory
	current_dir = os.path.dirname(os.path.abspath("__file__"))
	path_and_file = os.path.join(current_dir, data_path)
	# open the file to read the data
	f = open(path_and_file, "r")
	raw_data = [line.strip().split('\t') for line in f.readlines()]
	raw_data = np.array(raw_data, dtype = float)
	# attributes starts column 2 to the end, represent gene's expression values
	attributes = raw_data[:,2:]
	# ground_truth: column 1
	ground_truth = raw_data[:, 1]
	ground_truth = ground_truth.astype(int)
	# index: column 0
	index = raw_data[:, 0]
	index = index.astype(int)
	return attributes, ground_truth, index

def createDistMatrix(num_objs, data):
    # create the original matrix about distance, if no edge between two nodes, put float.max value as infinity number
    dist_matrix = [[sys.float_info.max for col in range(num_objs)] for row in range(num_objs)]
    dist_matrix = np.array(dist_matrix)
    for i in range(num_objs):
        for j in range(i + 1, num_objs):
            dist_matrix[i][j] = distance.euclidean(data[i], data[j])
    return dist_matrix

def build_heap(dist_matrix):
	# a, b are nodes, due to the index in original data, a and b value here is off by 1
	heapForDist_pair_nodes = []
	for a in range(dist_matrix.shape[0]):
		for b in range(dist_matrix.shape[1]):
			distance = dist_matrix[a][b]
			heapForDist_pair_nodes.append((distance, [distance, [a, b]]))
	# convert dist_pair_nodes to priority queue
	heapq.heapify(heapForDist_pair_nodes)
	return heapForDist_pair_nodes

def validate_nodes_NOTInSameClustering(left, right,clusterings):
	# if left and right node are both in the same clustering, then return false
	for clustering in clusterings:
		if (left in clustering) and (right in clustering):
			return False
	return True

def locateClustering(node, clusterings):
	for clustering in clusterings:
		if node in clustering:
			return clusterings.index(clustering)		

	return -1
	
def hierarchical_algorithm(k, heapForDist_pair_nodes):

	while (k != len(predict_clusterings)):
		try:
			dist, elems = heappop(heapForDist_pair_nodes)
			pair_data = elems[1]
			a = pair_data[0]
			b = pair_data[1]
			# check the node if not exist in the same cluster
			if validate_nodes_NOTInSameClustering(a,b,predict_clusterings):
				a_idx_clustering = locateClustering(a, predict_clusterings)
				b_idx_clustering = locateClustering(b, predict_clusterings)
				# concatenate two sets
				predict_clusterings[a_idx_clustering] += predict_clusterings[b_idx_clustering]
				# after concatenate two sets, you need to delete one set from the predict_clusterings
				predict_clusterings.remove(predict_clusterings[b_idx_clustering])
				# print ("current_cluster is: {}".format(predict_clusterings))

				# print ("the index of {} is {}".format(a, a_idx_clustering))
				# print ("the index of {} is {}".format(b, b_idx_clustering))
		except:
			break

			"""
			newNode = []
			idx_a = getClustering(a, clusterings)
			idx_b = getClustering(b, clusterings)
			if idx_a is None and idx_b is None:
				temp = []
				newNode.append(a)
				newNode.append(b)
				temp.append(a)
				temp.append(b)
				clusterings.append(temp)
				# delete the a and b from the S, become [a, b] pair
				S.remove([a])
				S.remove([b])
				S.append(newNode)
			elif idx_a is None and idx_b is not None:
				elem = clusterings[idx_b]
				S.remove(elem)
				clusterings[idx_b].append(a)
				S.append(clusterings[idx_b])
			elif idx_a is None and idx_b is not None:
				elem = clusterings[idx_a]
				S.remove(elem)
				clusterings[idx_a].append(b)
				S.append(clusterings[idx_a])
			"""
				# clusterings[idx_b] = clusterings[idx_b] + clusterings[idx_a]

				

			# clusterings[idx_b] = clusterings[idx_b] + clusterings[idx_a]
			# del clusterings[idx_a]

	return predict_clusterings

def organizeClusterings():
	for i in range(len(predict_clusterings)):
		predict_clusterings[i].sort()

	for i in range(len(predict_clusterings)):
		predict_clusterings[i] = [ (j + 1) for j in predict_clusterings[i]]



def getPredictLabels(num_objs):

	true_clusterings_labels = (num_objs,)
	true_clusterings_labels = np.zeros(true_clusterings_labels, dtype=int)

	for idx in range(len(predict_clusterings)):
		clustering = predict_clusterings[idx]
		for node in clustering:
			true_clusterings_labels[node - 1] = idx
    		
	return true_clusterings_labels


def calc_jaccard_rand(ground_truth, cluster_list):
	m00 = 0
	m01 = 0
	m10 = 0
	m11 = 0

	for i in range(len(ground_truth)):
		for j in range(len(cluster_list)):
			if (ground_truth[i] != ground_truth[j]) and (cluster_list[i] != cluster_list[j]):
				m00 += 1
			elif (ground_truth[i] != ground_truth[j]) and (cluster_list[i] == cluster_list[j]):
				m01 += 1
			elif (ground_truth[i] == ground_truth[j]) and (cluster_list[i] != cluster_list[j]):
				m10 += 1
			elif (ground_truth[i] == ground_truth[j]) and (cluster_list[i] == cluster_list[j]):
				m11 += 1
	
	jacc = float(m11) / float(m11 + m10 + m01)
	rand = float(m11 + m00) / float(m11 + m10 + m01 + m00)
	print("Jaccard is:",jacc)
	print("Rand is:", rand)


def doPCA(data, clusterings_labels):
	# convert matrix to dataframe
	DataMat = pd.DataFrame(data)

	pca = PCA(n_components=2)
	df_transform = pca.fit_transform(DataMat)
	labels = np.asarray(clusterings_labels).reshape(len(clusterings_labels),1)
	mat_labels = np.append(df_transform, labels, axis=1)
	final_df = pd.DataFrame(mat_labels)
	final_df.columns = ['x','y','label']
	final_df.label = final_df.label.astype(int)
	return final_df


def plotScatter(final_df,filename):
    groups = final_df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
    ax.set_title(filename)   
    ax.legend()
    
    fig_size = [12,9]
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")

    plt.show()
    fig.savefig("{}.png".format(filename))




if __name__ =="__main__":

	k = 0
	inputFile = sys.argv[1]
	try:
		K = sys.argv[2]
	except:
		print ("\n correct input args are:\n python3 min.py fileName k_clustering")
		Y_N = input("Are you sure not define the K for clustering? [Y/N]")
		if Y_N == "n" or Y_N == "n":
			K = input("Pls input K: ")
			k = int(K)
	"""
	---------------------------------------Step1 : load the data-------------------------------------------
	"""
	# load the data
	attributes, true_clusterings_labels, gene_id = load_data(inputFile)
	default_num_clustering = len(set(true_clusterings_labels))
	# get number of objs
	num_objs = len(attributes)
	
	try:
		k = int(K)
	except:
		k = default_num_clustering

	if k != default_num_clustering:
		if k > num_objs:
			print("K has to be smaller than default clustering size, we will default it")
			k = default_num_clustering

	# create the original matrix about distance, if no edge between two nodes, put float.max value as infinity number
	original_dist_matrix = createDistMatrix(num_objs, attributes)

	"""
	------------------------------------Step 2: hierarchical_algorithm --------------------------------------
	"""

	heapForDist_pair_nodes = build_heap(original_dist_matrix)
	# print (heapForDist_pair_nodes)
	global predict_clusterings
	predict_clusterings = [[node] for node in range(num_objs)]

	hierarchical_algorithm(k, heapForDist_pair_nodes)

	organizeClusterings()

	"""
	------------------------------------Step 2: Print Single link --------------------------------------
	"""
	print("\nfinal clustering:")
	i = 1
	for clustering in predict_clusterings:
		print ("clustering{} has {} items ---- {}".format(i, len(clustering), clustering))
		i += 1

	"""
	------------------------------------Step 3 : Calculate Jaccard and Rand --------------------------------------
	"""
	predict_clusterings_labels = getPredictLabels(num_objs)

	calc_jaccard_rand(true_clusterings_labels, predict_clusterings_labels)

	"""
	------------------------------------Step 4 : PCA --------------------------------------
	"""
	final_df = doPCA(attributes,predict_clusterings_labels)
	plotScatter(final_df,"PCA plot for " + inputFile + "\n k = " + str(k))
