import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def initialize_means(k:int)->np.array:
	#intialize the centers
	indexes = np.random.choice(range(n), k, replace = False)
	return(indexes)

class KMeans():
	def __init__(self, k:int, epsilon:float = 10e-9, centers:np.array=None):
		self.k = k
		self. epsilon = epsilon
		self.centers = centers

	def fit(self, data:np.array)->None:
		#data : n x d
		#centers: k x d
		if self.centers is None:
			centers_updated = data[initialize_means(k),:]
		else:
			centers_updated = self.centers.copy()
		centers = 2*centers_updated.copy()
		#reshape data for broadcasting
		data_res = np.reshape(data,(n,1,d))
		#until convergence
		while not (np.allclose(centers,centers_updated, rtol = self.epsilon)):
			centers = centers_updated.copy()
			predictions = np.argmin(np.linalg.norm(data_res - centers, axis = 2), axis = 1)
			for c in range(k):
				mask = (predictions==c)
				n_class = sum(mask)
				avg_class = np.sum(data[mask,:], axis = 0)
				centers_updated[c] = avg_class/n_class
		self.centers = centers_updated.copy()

	def predict(self, x:np.array)->int:
		pred = np.argmin(np.linalg.norm(self.centers-x.T, axis = 1))
		return(pred)


if __name__=="__main__":
	n = 90
	d = 2
	k = 3
	actual_centers = np.array([(1,2),(1,0),(3,1)])
	print("Actual Centers:")
	print(actual_centers)
	#creating a toy dataset
	training_data= np.array([[np.random.normal(loc = i, scale = 0.1), np.random.normal(loc = j, scale = 0.1)] for i,j in actual_centers for z in range(n//k)])
	df = pd.DataFrame(training_data, columns =["x","y"])
	labs = np.repeat(list(range(k)), np.repeat(n//k,k))
	df["labels"]= labs
	#visualizing
	sns.scatterplot(x = "x", y = "y", hue = "labels",data = df, palette ="bright")
	plt.show()
	#init
	knn = KMeans(k = k, centers = None)
	#training
	knn.fit(training_data)
	print("Predicted Centers:")
	#they might be in different order
	print(knn.centers)
	sns.scatterplot(x = "x", y = "y", hue = "labels",data = df, palette ="bright")
	plt.scatter(x = knn.centers[:,0],y =  knn.centers[:,1], marker = "o", s = 100, color = "k")
	plt.show()


