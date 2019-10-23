# PrincipalComponentAnalysis-PCA-
In this Project the aim is to analyze breast cancer wisconsin dataset from sklearn &amp; perform PCA(Principal Component Analysis) on this dataset. 1. Load the breast cancer dataset from sklearn datasets package.  2. Store data &amp; it's class in separate numpy arrays.  3. Perform PCA on all 30 dimensions of data using PCA functions in sklearn decomposition package.  4. Plot  graph with cumulative explained variance against number of components.  5. By analyzing this graph find value of reduced number of components such that cumulative variance is maximized &amp; number of principal components is minimized.  6. Calculate explained_variance, explained_variance_ratio &amp; cumulative sum of variance.  7. Plot a scatterplot for top 2 principal components such that each class is represented by a separate color. Verify if the classes benign &amp; malignant are separable in plot or not.  8. Plot a 3D scatterplot for top 3 principal components, also plot two separate 2D plots among PCA1, PCA3 &amp; PCA2, PCA3.  sklearn PCA documentation for reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
