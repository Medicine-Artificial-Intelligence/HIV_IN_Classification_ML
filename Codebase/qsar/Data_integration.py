import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split



class Data_Integration():
    """
    Create Data Frame from csv file, find missing value (NaN), choose a threshold to make target transformation (Classification)
    remove handcrafted value (Regression), split Data to Train and Test and show the chart
    
    Input:
    -----
    data : pandas.DataFrame
        Data with features and target columns
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
   
        
    Returns:
    --------
    Data_train: pandas.DataFrame
        Data for training model
    Data_test: pandas.DataFrame
        Data for external validation  
    """
    def __init__(self, data, activity_col, task_type, target_thresh):
        
        self.data = data
        self.activity_col= activity_col
        self.task_type = task_type
        self.target_thresh = target_thresh
        
            
        
        
        
    # 1. Check nan value - Mark Nan value to np.nan
    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan 
    
    # 2. Target transformation - Classification
    def target_bin(self, thresh, input_target_style = 'pIC50'):
        if input_target_style != 'pIC50':
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 1
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 0
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        else:
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 0
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 1
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        
    
    # 3. Split data
    #Chia tập dữ liệu  thành tập train và test.
    def Data_split(self):
        
        if self.task_type.title() == "C":
            if len(self.data[self.activity_col].unique()) ==2: 
                y = self.data[self.activity_col]
            else:
                self.target_bin(thresh = self.target_thresh)
                y = self.data[self.activity_col]
            
            self.stratify = y
        
        elif self.task_type.title() == "R":
            y = self.data[self.activity_col]
            self.stratify = None
            
        
       
        X = self.data.drop([self.activity_col], axis =1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, 
                                                            random_state = 42, stratify = self.stratify)


        #index:
        self.idx = X.T.index

        #Train:
        self.df_X_train = pd.DataFrame(X_train, columns = self.idx)
        self.df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
        self.data_train = pd.concat([self.df_y_train, self.df_X_train], axis = 1)
        

        #test
        self.df_X_test = pd.DataFrame(X_test, columns = self.idx)
        self.df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
        self.data_test = pd.concat([self.df_y_test, self.df_X_test], axis = 1)
        
        print("Data train:", self.data_train.shape)
        print("Data test:", self.data_test.shape)
        print(75*"*")
        

    def Visualize_target(self):
        if self.task_type.title() == "C":
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.title(f'Training data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_train.iloc[:,0])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_train.iloc[:,0].values == 1).sum() / (self.data_train.iloc[:,0].values == 0).sum(),3))}')
            plt.subplot(1,2,2)
            plt.title(f'External data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_test.iloc[:,0])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_test.iloc[:,0].values == 1).sum() / (self.data_test.iloc[:,0].values == 0).sum(),3))}')
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
        else:
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            sns.histplot(self.data_train.iloc[:,0], palette = 'deep', kde = True)
            #plt.hist(self.Data_train.iloc[:,0])
            plt.title(f'Train set distribution', weight = 'semibold', fontsize = 16)
            plt.subplot(1,2,2)
            #plt.hist(self.Data_test.iloc[:,0])
            sns.histplot(self.data_test.iloc[:,0], palette = 'deep', kde = True)
            plt.title(f'External validation set distribution',weight = 'semibold', fontsize = 16)
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
        
         
    def fit(self):
        self.data.apply(self.Check_NaN)
        self.Data_split()
        self.Visualize_target()