'''
Created on 14 Aug 2018

@author: Gavin Smith
@organization: N/LAB, The University of Nottingham.

@copyright: This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

import rpy2
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()

importr('party')
importr('edarf')

class pycforest(object):
    '''
    An implementation of the random forest and bagging ensemble algorithms utilizing conditional
    inference trees as base learners.
    
    This is a wrapper for cforest within the party R package.
    For more details on the algorithm see:
    https://cran.r-project.org/web/packages/party/party.pdf
    page 6 & 7.
    
    An important academic reference is:
    Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis and Torsten Hothorn (2007). Bias in Random
    Forest Variable Importance Measures: Illustrations, Sources and a Solution. BMC Bioinformatics,
    8, 25. http://www.biomedcentral.com/1471-2105/8/25
    
    Variable importance is done via the R edarf package.
    See the Reference manual and vignettes at: 
    https://cran.r-project.org/web/packages/edarf/index.html
    for more details.
    
    NOTES: Factors must be encoded as str in the pandas DataFrame. Coding as Category dtype is NOT enough.
    
    INSTALATION NOTES: 
        * You must have R installed and the packages 'party' and 'edarf' installed within R.
    
    Todo:
        * Place a check on the incoming pandas DataFrames to ensure Categories are encoded as str and fix
          them if not.
        * Add better documentation.
    '''


    def __init__(self, n_trees = 500, max_features = None, maxsurrogate = None, debug = False):
        '''
        Constructor
        
        Args:
            n_trees (int):              Number of trees to train within the forest.
            max_features (int or None): The number of features to consider when looking for the best split. 
                                        Translated to mtry for the underlying R call.
                                        If None, then max_features=n_features.
            maxsurrogates (int or None):number of surrogate splits to evaluate. Note the currently only surrogate splits
                                        in ordered covariables are implemented.
                                        If None, all possible surrogates will be considered.
            debug (bool):               If true, display the types of the R dataframe as converted within
                                        the fit method.

        Returns:
            bool: The return value. True for success, False otherwise.
        '''
        self.debug = debug
        self.mtry = max_features
        self.n_trees = n_trees
        self.replace = False
        self.maxsurrogate = maxsurrogate
        self.r_model = None
        self.fitct = 0
        self.oobpredct = 0
        self.oob_prediction_accuracy = None
        self.forest_type_is_classification = None
    
    
    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).
        Args:
            X (pandas DataFrame, [n_samples, n_features]): The training input samples. Categorical features should be encoded as str. 
                                                           These will be converted to R factors.

            y (pandas DataFrame, [n_samples])]:             The output labels. If this is a classification problem these MUST be 
                                                            encoded as str.
        """
        data = X.assign(labels = y)
        self.X = X
        self.y = y
        r_data = pandas2ri.py2ri(data)
        self.r_data =r_data
        
        if self.debug:
            print("Data's types in R:")
            print(r.sapply(r_data, 'class'))
        
        r('pclass <- class')
        if r.pclass(r_data[-1])[0] == "factor":
            self.forest_type_is_classification = True
        
      
        if self.mtry == None:
            self.mtry = X.shape[1] / 2
        
        if self.maxsurrogate == None:
            self.maxsurrogate = X.shape[1] -1 
        
        r('as_formula <- as.formula')
        self.r_model = r.cforest(formula=r.as_formula(r.paste('labels', r.paste(X.columns, collapse=" + "), sep=" ~ ")), data=r_data, control = r.cforest_unbiased(mtry = self.mtry, ntree=self.n_trees, maxsurrogate = self.maxsurrogate))
        
        self.fitct += 1
        
      
    
    
    def predict(self, X = None):
        """
        Predicts values for the given input features
        
        Args:
            n_trees (int):              Number of trees to train within the forest.
            max_features (int or None): The number of features to consider when looking for the best split. 
                                        Translated to mtry for the underlying R call.
                                        If None, then max_features=n_features.
            maxsurrogates (int or None):number of surrogate splits to evaluate. Note the currently only surrogate splits
                                        in ordered covariables are implemented.
                                        If None, all possible surrogates will be considered.
            debug (bool):               If true, display the types of the R dataframe as converted within
                                        the fit method.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        if self.r_model is None:
            raise Exception('Model must be trained first.') 
        
        if X is None:
            r_rtn = r.predict(self.r_model, OOB=True)  
            
        else:
            r_X = pandas2ri.py2ri(X)
            r_rtn = r.predict(self.r_model, newdata = r_X)  
        

        if self.forest_type_is_classification:  
            rtn = np.asarray([ r_rtn.levels[r_rtn[x] - 1] for x in r_rtn])
        else:
            rtn = np.asarray([ x for x in r_rtn])
        
         
        
        return np.asarray(rtn)
    
    def get_oob_prediction_accuracy(self):
        
        """
        Computes and returns the out-of-bag prediction accuracy for the random forest.
        
        Returns:
        
        The mean out-of-bag prediction accuracy.
        
        """
        
        
        if self.r_model is None:
            raise Exception('Model must be trained first.') 
        
        r('as_vector <- as.vector')
        if self.oob_prediction_accuracy is None or self.fitct -1 > self.oobpredct:
           
            self.oob_prediction_accuracy = np.mean(self.y.values.flatten() == self.predict())
            
            self.oobpredct += 1
        
        return self.oob_prediction_accuracy
        
    def permutation_importance(self, interaction = False, oob = True, vars = None, type = "aggregate", nperm = 10, gav_method = True):
        
        """
        Computes the permutation importance.
        
        Args:
            interaction (bool):              
            oob (bool): 
                                        
                                        
            vars (list of variable names or None):
                                       
                                       
            type (str):              
                                        

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        
        
        if self.r_model is None:
            raise Exception('Model must be trained first.') 
        
        if vars is None:
            vars = self.X.columns
                  
        rtn = r.variable_importance(self.r_model, var = vars, type = type, nperm = nperm, oob=oob, interaction = interaction, gav_method = gav_method)
         
        df_p = pandas2ri.ri2py(rtn)
        
        df = pd.DataFrame(df_p,columns=["importance"])
        df = df.assign(factors = vars)
        
        return df
        
    
if __name__ == '__main__':
    rf = RTree2(n_trees=500)
    X = pd.read_csv("test_X.csv").iloc[0:100,:]
    y = pd.read_csv("test_y.csv").iloc[0:100,:]
    y['labels'] = y['labels'].astype(str)
  
    
    rf.fit(X,y)
    print(rf.predict(X))
    print(y)
#     print(rf.get_oob_prediction_accuracy())
#     print(rf.permutation_importance())
    