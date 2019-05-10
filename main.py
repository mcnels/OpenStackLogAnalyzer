import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import scipy.spatial.distance as ssd


'''

    COLLECTION AND NORMALIZATION

'''

# From an array containing the messages, get a term document data frame
def getTermDocDataFrame(docs, xColNames = None, **kwargs):
    #initialize the vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)

    #create dataFrame and using words as index for the rows
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames

    # return the data frame with words as rows
    return df

# read log messages from file
dataset = pd.read_csv('testdata.txt', sep=" ")

# Keep Message column from dataset
dataset1 = dataset.Message

# declare an array to hold the normalized messages
dataset2 = []

# remove stopwords from the dataset (the, a, an, etc.)
stopwordsList = set(stopwords.words('english'))

# Pre-processing: remove numbers, lower the case, remove punctuation, remove stopwords and replace paths
for data in dataset1:
    data1 = re.sub(r'\d+', '', data.lower())
    data1 = re.sub('instance: (.+?)]', 'instance', data1)
    data1 = re.sub("directory '(.*?)'", 'directory PATH', data1)
    data1 = re.sub('['+string.punctuation+']', '', data1)
    data1 = [w for w in word_tokenize(data1) if not w in stopwordsList]
    dataset2.append(str(data1).strip())

# Print out split messages, listing out the words composing each message
print('-----Tests-----')
print(dataset2[0]) # first message
print(dataset2[-1]) # last message
print()

# Generate term document data frame from the set of message errors
print('-----Term document data frame-----')
termDoc = getTermDocDataFrame(dataset2)
print(termDoc)
print()

# Generate a matrix out of the data frame
# The rows of that matrix represent unique words in the whole dataset
# The columns represent the messages
print('-----Term document matrix-----')
termDocMatrix = np.asarray(termDoc)
print(termDocMatrix)
print()

# Compute distance matrix
# distance between messages (using the transpose of the termDocMatrix)
distMatrix = euclidean_distances(termDocMatrix.T)
print('-----Distance matrix-----')
print(distMatrix)
print()

# convert the redundant n*n square matrix form into a condensed nC2 array
# Have to use a 1D condensed matrix with shc.linkage function
print('-----Condensed Distance matrix-----')
distArray = ssd.squareform(distMatrix)
print(distArray)
print()


'''

    CLUSTERING AND DENDROGRAM

'''

# Plot Cluster Dendrogram using scipy
plt.figure(figsize=(15, 15))
plt.title("Messages Dendrogram with scipy")
dend = shc.dendrogram(shc.linkage(distArray, method='single'))
plt.show()

# Plot dendrogram to be used with sklearn
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    shc.dendrogram(linkage_matrix, **kwargs)

# Perform agglomerative clustering with sklearn
model = AgglomerativeClustering(n_clusters=5, linkage='single')
model.fit(termDocMatrix.T)
plt.title("Messages Dendrogram with sklearn")
plot_dendrogram(model)
plt.show()


'''

    CORRELATION BETWEEN CLUSTERS USING DTW

'''

# Import DTW library
from dtaidistance import dtw

# Get time information from messages dataset
timeSeries = dataset.DeviceReportedTime
timeSeriesArray = np.asarray(timeSeries)

# Remove duplicate time values and keep order
nodupTS = []
[nodupTS.append(item) for item in timeSeriesArray if item not in nodupTS]

# Obtain the clusters from linkage matrix used to plot the dendrogram
# 1.6 is the cut off point on y-axis of the dendrogram (represents the distance between the clusters)
clusters = shc.fcluster(shc.linkage(distArray, method='single'), 1.6, 'distance')

# preparing time column to insert in the data frame (a time for each msg)
timeCol = []
for msg in dend['leaves']:
    timeCol.append(timeSeriesArray[int(msg)])

# list containing unique date times sorted in chronological order
timeSorted = sorted(nodupTS)

# assign an ID to each time value of a message, based on the index of the unique time values
timeID = []
for time in timeCol:
    for id in timeSorted:
        # if a message's occurrence time coincides with a time in the array containing unique time values, assign the
        # index of the unique time value as the time ID for the message
        if str(time) == str(id):
            timeID.append(timeSorted.index(time))

# Data frame containing each message, the corresponding cluster, time occurred and time ID
cluster_output = pd.DataFrame({'msgID':dend['leaves'], 'cluster':clusters, 'time':timeCol, 'timeID': timeID})
print("--------Cluster Output--------")
print(cluster_output)
print()


def extractMsgFromCluster(allClusters, id):
    # Extract each cluster from the cluster data frame
    cl = allClusters.loc[allClusters['cluster'] == id]
    # Sort messages in chronological order
    cl = cl.sort_values(by=['timeID'])
    # Drop duplicate time occurrences, keeping only the first occurrence
    cl = cl.drop_duplicates(subset='timeID', keep='first')
    # make timeID the index of the messages for later use in the dtwInput data frame
    cl = cl.sort_values(by=['timeID']).set_index('timeID')
    return cl['msgID']

# Extracting the message IDs that corresponding to each unique time occurrence, from each cluster (4 clusters in this case)
nc1 = extractMsgFromCluster(cluster_output, 1)
nc2 = extractMsgFromCluster(cluster_output, 2)
nc3 = extractMsgFromCluster(cluster_output, 3)
nc4 = extractMsgFromCluster(cluster_output, 4)

# Data frame representing clusters as columns and rows as unique time occurrences found in the dataset
# Each (time, cluster) represents the message ID from said cluster, received at said time
# This data frame is used later as input to the DTW algorithm
dtwInput = pd.DataFrame({'cluster 1':nc1, 'cluster 2':nc2, 'cluster 3':nc3, 'cluster 4':nc4})
print("--------DTW Input--------")
print(dtwInput)
print()

# Labeling each of the clusters
x = np.array(nc1)
y = np.array(nc2)
c = np.array(nc3)
d = np.array(nc4)

# Plot each cluster on a scatter plot
plt.figure(figsize=(15, 15))
plt.title("Clusters")
plt.plot(x,'r', label='x')
plt.plot(y, 'g', label='y')
plt.plot(c,'b', label='c')
plt.plot(d, 'y', label='d')
plt.show()

# Generate the data frame representing the messages appearing in each cluster
# at a particular time. Each row represents a cluster with 'message ID' elements
# Columns represent time values in chronological order
print("-----Date matrix-----")
dateMatrix = np.asarray(dtwInput)
# Transpose to get clusters as rows instead of clusters
print(dateMatrix.T)
print()

# Distance matrix using DTW algorithm. Transpose used because we want distance between clusters(rows)
ds = dtw.distance_matrix(dateMatrix.T)
print('-----Distance matrix from DTW-----')
print(ds)
print()
