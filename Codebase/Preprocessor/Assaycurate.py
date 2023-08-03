import numpy as np
import pandas as pd
import seaborn as sns


class assay_curate:
    def __init__(self,data, type_col, org_col,des_col, type_arg='F', org_arg='Homo sapiens', kw = 'MTT'):
        self.data = data
        self.type_col = type_col
        self.org_col = org_col
        self.des_col = des_col
        self.type_arg= type_arg
        self.org_arg= org_arg
        self.kw = kw 
    
    def search_kw(self, data,kw, des_col):
        index = []
        for key, value in enumerate(data[des_col]):
            if kw in value:
                index.append(key)
        return data.iloc[index,:]
    
    def curated_fit(self):
        print("Number of data befor standardizing:", self.data.shape[0])
        df = self.data[self.data[self.type_col]==self.type_arg]
        print("Number of data after choosing assay type:", df.shape[0])
        df2 = df[df[self.org_col]==self.org_arg]
        print("Number of data after choosing assay organism:", df2.shape[0])
        df3 = self.search_kw(data=df2, kw = self.kw, des_col = self.des_col)
        print("Number of data after curating:", df3.shape[0])
        self.df = df3
