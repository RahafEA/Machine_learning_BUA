"""
b) Identify which species each target number (0, 1, 2) represents
Each species.
c) Analyze the data using Pandas.
d) Visualize the relationship between Sepal Length and Sepal Width.
e) Calculate the average sepal length for each of the three
species.
1 a) Modify the scatter plot to visualize petal length vs. petal
width.
2 a) Create a histogram for a single feature (e.g., petal width).
What does the distribution tell you about the data?

"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import datasets 
 
iris = datasets.load_iris() 
x = iris.data 
y = iris.target 
print(iris.target_names)   

iris = pd.DataFrame( 
    data=np.c_[iris['data'], iris['target']], 
    columns=iris['feature_names'] + ['target'] 
) 
#مكنتش عملاه بس عشان الرسم يظبط
iris['species'] = iris['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(iris.head()) 
print(iris.info()) 
print(iris.describe()) 
 
# Scatter 
plt.figure(figsize=(5,5)) 
plt.scatter(iris["sepal length (cm)"], iris["sepal width (cm)"], c=iris["target"], cmap='viridis', edgecolor='k') 
plt.xlabel("sl") 
plt.ylabel("Sw") 
plt.title("Sepal Length and Sepal Width") 
plt.show() 
 
#scatter 
plt.figure(figsize=(5,5)) 
plt.scatter(iris["petal length (cm)"], iris["petal width (cm)"], c=iris["target"], cmap='plasma', edgecolor='k') 
plt.xlabel("Pl") 
plt.ylabel("Pw") 
plt.title("Petal Length and Petal Width") 
plt.show() 
 
# Histogram 
plt.figure(figsize=(6,4)) 
plt.hist(iris["petal width (cm)"], bins=15, edgecolor='black') 
plt.xlabel("Pw") 
plt.ylabel("التغير") 
plt.show() 
 
avg = iris.groupby("species")["sepal length (cm)"].mean()   
print(avg)
