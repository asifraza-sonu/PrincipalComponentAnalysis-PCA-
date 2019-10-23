import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
breast_cancer_data = datasets.load_breast_cancer()
#breast_cancer_data
X = breast_cancer_data.data
y = breast_cancer_data.target
X.shape
pca1 = PCA(30)
pca2 = pca1.fit(X)
print(pca2.explained_variance_ratio_.cumsum())
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative Explained variance')
plt.savefig('Cumulative Explained variance.png')
plt.show()
plt.close()
#By analysing the above figure we can see there is not much variance percentage change after the number of components 2 and after 3
#Reducing components further to 2 and visualize resulting principal components
pca3 = PCA(n_components=2)
pca4 = pca3.fit_transform(X)
pca4
#2DPlot for PCA1 and PCA2 
plt.figure(figsize=(15,10))
plt.scatter(pca4[:,0], pca4[:,1], c=y, edgecolor='none', alpha=0.7,
           cmap=plt.get_cmap('jet', 10), s=20)
plt.colorbar()
plt.xlabel('pca one')
plt.ylabel('pca two')
plt.savefig('2D Plot_PCA1_PCA2.png')
plt.show()
plt.close()
#For first three principal components-3d plot
pca5 = PCA(n_components=3)
pca6 = pca5.fit_transform(X)
pca6
ax = plt.figure(figsize=(16,12)).gca(projection='3d')
ax.scatter(
    xs=pca6[:,0], 
    ys=pca6[:,1], 
    zs=pca6[:,2], 
    c=y, 
    cmap='tab10'
)
ax.set_xlabel('pca one')
ax.set_ylabel('pca two')
ax.set_zlabel('pca three')
plt.savefig('3 pri_comp_3D.png')
plt.show()
plt.close()
#2DPlot for PCA1 and PCA3 
plt.figure(figsize=(15,10))
plt.scatter(pca6[:,0], pca6[:,2], c=y, edgecolor='none', alpha=0.7,
           cmap=plt.get_cmap('jet', 10), s=20)
plt.colorbar()
plt.xlabel('pca one')
plt.ylabel('pca three')
plt.savefig('2D Plot_PCA1_PCA3.png')
plt.show()
plt.close()
#2DPlot for PCA2 and PCA3 
plt.figure(figsize=(15,10))
plt.scatter(pca6[:,1], pca6[:,2], c=y, edgecolor='none', alpha=0.7,
           cmap=plt.get_cmap('jet', 10), s=20)
plt.colorbar()
plt.xlabel('pca two')
plt.ylabel('pca three')
plt.savefig('2D Plot_PCA2_PCA3.png')
plt.show()
plt.close()
#For first two principle components PCA1 and PCA2 - Explained Variance
pca7 = PCA(n_components=2)
pca8 = pca7.fit(X)
#Checking explained variance for PCA1 and PCA2
print('For PCA1 and PCA2 the Explained Variance is :',pca8.explained_variance_)
print('For PCA1 and PCA2 the Explained Variance ratio is :',pca8.explained_variance_ratio_)
print('For PCA1 and PCA2 the cumulative Explained Variance is :',pca8.explained_variance_ratio_.cumsum())
#For first three principle components PCA1 ,PCA2,PCA3 - Explained varinace
pca9 = PCA(n_components=3)
pca10 = pca9.fit(X)
#Checking explained variance for PCA1,PCA2 and PCA3
print('For PCA1 PCA2  and PCA3 the Explained Variance is :',pca10.explained_variance_)
print('For PCA1 PCA2  and PCA3 the Explained Variance ratio is :',pca10.explained_variance_ratio_)
print('For PCA1 PCA2  and PCA3 the cumulative Explained Variance is :',pca10.explained_variance_ratio_.cumsum())