#Singular Random Forest Hyperoptimisation Model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class BaseTMLModels(BaseEstimator, RegressorMixin):

  def __init__(self,species = [], n_estimators = 100, criterion = 'squared_error',
   max_features =1, min_samples_split =2, min_samples_leaf = 1, bootstrap = False, min_weight_fraction_leaf = 0, 
   max_depth = None, max_leaf_nodes= None, min_impurity_decrease =0, random_state = 0 ):
      """
      A Custom BaseEstimator that can switch between classifiers.
      :param estimator: sklearn object - The classifier
      """ 
      self.species = species
      
      self.n_estimators = n_estimators
      self.criterion = criterion
      self.max_features = max_features
      self.min_samples_split = min_samples_split
      self.min_samples_leaf = min_samples_leaf
      self.bootstrap = bootstrap
      self.min_weight_fraction_leaf = min_weight_fraction_leaf
      self.max_depth = max_depth
      self.max_leaf_nodes = max_leaf_nodes
      self.min_impurity_decrease = min_impurity_decrease
      self.random_state = random_state

  
  def fit(self, X, y=None, **kwargs):
      self.species_dict_ = {}
      #print(X.shape, y.shape)
      for s in self.species: 
          x_s = X.loc[X['Test organisms (species)'] == s]
          if x_s.shape[0] > 0:
              self.species_dict_[s] = RandomForestRegressor( n_estimators = self.n_estimators, criterion= self.criterion, max_features= self.max_features,
                                                             min_samples_split= self.min_samples_split, min_samples_leaf = self.min_samples_leaf, bootstrap = self.bootstrap,
                                                              min_weight_fraction_leaf = self.min_weight_fraction_leaf, max_depth = self.max_depth, max_leaf_nodes = self.max_leaf_nodes,
                                                               min_impurity_decrease = self.min_impurity_decrease, random_state =self.random_state)
              x_s = x_s.drop(columns = ['Test organisms (species)'])
              y_s = y.loc[x_s.index.tolist()]
              self.species_dict_[s].fit(x_s,y_s)
      return self
  
  
  def predict(self, X, y=None):
      X.reset_index(drop = True, inplace = True)
      y_result = np.empty((X.shape[0],))
      for s in X['Test organisms (species)'].unique().tolist():
          x_s = X.loc[X['Test organisms (species)'] == s]
          x_s = x_s.drop(columns = ['Test organisms (species)'])
          if s in self.species_dict_.keys() and x_s.shape[0]> 0:
              temp_pred = self.species_dict_[s].predict(x_s)
              j = 0
              for i in x_s.index.tolist():
                  y_result[i] = temp_pred[j] 
                  j +=1
      return y_result
  
  def score(self, X, y):
      X.reset_index(drop= True, inplace = True)
      y.reset_index(drop= True, inplace = True)
      y_result = np.empty(y.shape)
      for s in self.species:
          x_s = X.loc[X['Test organisms (species)'] == s]
          if x_s.shape[0] > 0 and s in self.species_dict_.keys():
            x_s = x_s.drop(columns = ['Test organisms (species)'])
            temp_pred = self.species_dict_[s].predict(x_s)
            j = 0
            for i in x_s.index.tolist():
                y_result[i] = temp_pred[j] 
                j +=1
      result = -metrics.mean_squared_error(y, y_result, squared = False)
      return result