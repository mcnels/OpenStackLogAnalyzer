import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def fn_tdm_df(docs, xColNames = None, **kwargs):
    #initialize the  vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames

    return df

# read data from file
dataset = pd.read_csv('logdata2.tr', sep=" ")

# Keep Message column from dataset
dataset1 = dataset.Message
dataset2 = []

stopwordsList = set(stopwords.words('english'))

for data in dataset1:
    # remove numbers, lower the case, remove punctuation, remove stopwords and replace paths
    data1 = re.sub(r'\d+', '', data.lower())
    data1 = re.sub('instance: (.+?)]', 'instance', data1)
    data1 = re.sub("directory '(.*?)'", 'directory PATH', data1)
    data1 = re.sub('['+string.punctuation+']', '', data1)
    data1 = [w for w in word_tokenize(data1) if not w in stopwordsList]
    dataset2.append(str(data1).strip())

print('-----Tests-----')
print(dataset2[0])
print(dataset2[45304])
print(dataset2[45305])
print(dataset2[45306])
print(dataset2[45307])
print(dataset2[106450])
print(dataset2[-1])
print()

# Generate term document data frame from the set of message errors
print('-----Term document data frame-----')
termDoc = fn_tdm_df(dataset2)
print(termDoc)
print()

# Generate a matrix out of the data frame
print('-----Term document matrix-----')
termDocMatrix = np.asmatrix(termDoc)
print(termDocMatrix)
print()

# Compute distance matrix
print('-----Distance matrix-----')
distMatrix = euclidean_distances(termDocMatrix, termDocMatrix)
# distMatrix1 = distance.cdist(termDocMatrix, termDocMatrix, 'euclidean')
print(distMatrix)
print(distMatrix.shape)
print()
# print('***')
# print(distMatrix1)

# Plot Cluster Dendrogram
plt.figure(figsize=(13, 7))
plt.title("Messages Dendrogram")
dend = shc.dendrogram(shc.linkage(termDocMatrix, method='ward'))
plt.show()

# Hierarchical Clustering
#termDocArray = np.asarray(termDoc)
# cluster = AgglomerativeClustering(affinity='euclidean', linkage='ward')
# # cluster.fit_predict(distMatrix)
# cluster = cluster.fit(distMatrix)
# plt.figure(figsize=(10, 7))
# # dend1 = shc.dendrogram(shc.linkage(distMatrix, method='ward'))
# plt.scatter(distMatrix[:,0], distMatrix[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show()