#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *


# In[2]:


classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = load_iris()
classifier.fit(iris.data, iris.target)

viz = dtreeviz(classifier, 
               iris.data, 
               iris.target,
               target_name='variety',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
              
viz.view() 


# In[3]:


iris.feature_names


# In[6]:


iris.target


# In[ ]:




