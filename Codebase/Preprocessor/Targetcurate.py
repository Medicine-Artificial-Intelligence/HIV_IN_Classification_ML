import numpy as np
import pandas as pd
import seaborn as sns


class target_curate:
    def __init__(self, data, target_name_col, target_name, target_org_col, target_org,
                 type_col, unit_col, active_col, relate_col, type_arg,equal_only = False, thresh = 7):
        self.data = data.reset_index(drop=True)
        self.target_name_col = target_name_col 
        self.target_name = target_name
        self.target_org_col=target_org_col
        self.target_org = target_org
        self.type_col = type_col
        self.unit_col = unit_col
        self.active_col = active_col
        self.relate_col = relate_col
        self.type_arg = type_arg
        self.equal_only = equal_only
        self.thresh = thresh
        
    def target_filter(self, data, target_name_col, target_name, target_org_col, target_org):
        df= data[data[target_name_col]==target_name]
        display(df.shape)
        df2 = df[df[target_org_col]==target_org]
        display(df2.shape)
        return df2
    def standardize_value(self, data, type_col, type_arg, unit_col):
        df = data[data[type_col]==type_arg]
        df= df.dropna(subset =unit_col)
        df.reset_index(drop=True, inplace = True)
        type = ['μM','µM', 'nM', 'mM', 'M', 'nmol/l']
        idx = []
        for key, value in enumerate(df[unit_col]):
            if value in type:
                idx.append(key)
        df = df.iloc[idx,:]
        return df
    
    def convert_activity(self, data, active_col, unit_col):
        df = data.copy()
        df['pChEMBL'] = np.zeros(len(df))
        #unit = df['Unit'].unique()

        for key, value in enumerate(df[unit_col]):
            if value == 'μM':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1e-6)
            elif value  == 'µM':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1e-6)
            elif value  == 'nM':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1e-9)
            elif value  == 'nmol/l':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1e-9)
            elif value  == 'mM':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1e-3)
            elif value  == 'M':
                df.loc[key, 'pChEMBL'] = -np.log10(df.loc[key, active_col]*1)
            elif value  == 'no unit':
                df.loc[key, 'pChEMBL'] = -df.loc[key, active_col]
        return df
    
    
    def standardize_relation(self, data,relate_col,  equal_only, thresh):
        df = data.copy()
        df.dropna(subset = relate_col, inplace = True)
        if equal_only == True:
            print('SELECTING ONLY EQUAL')
            df = df[df[relate_col]=="'='"]
            
        else:
            print('HANDLING')
            df_big = df[(df[relate_col] == "'>'") | (df[relate_col] == "'>='")]
            df_small = df[(df[relate_col] == "'<'") | (df[relate_col] == "'<='")]
            df_equal = df[df[relate_col]=="'='"]
                
            #Drop pCHEMBL < thresh for df_big
            drop_idx = df_big[df_big["pChEMBL"] < thresh].index
            df_big.drop(drop_idx, inplace = True)
                
            #Drop pCHEMBL > thresh for df_small
            drop_idx = df_small[df_small["pChEMBL"] > thresh].index
            df_small.drop(drop_idx, inplace = True)
                
            df = pd.concat((df_equal, df_small, df_big), axis = 0)
        return df
                  
    def curated_fit(self):
        print("Number of data before target curation:", self.data.shape[0])
        df = self.target_filter(data = self.data, target_name_col = self.target_name_col, target_name =self.target_name, 
                           target_org_col=self.target_org_col, target_org = self.target_org)
        print("Number of data after handle organism and target name:", df.shape[0])
        df1 = self.standardize_value(data=df, type_col=self.type_col, type_arg=self.type_arg, unit_col=self.unit_col)
        df1.reset_index(drop=True, inplace = True)
        print("Number of data after select unit:", df1.shape[0])
        #display(df.head(5))
        df2 = self.convert_activity(data=df1, active_col=self.active_col, unit_col = self.unit_col)
        df2.reset_index(drop=True, inplace = True)
        #display(df2.head(5))
        df3 = self.standardize_relation(data=df2,relate_col=self.relate_col,  equal_only=self.equal_only, thresh=self.thresh)
        self.df = df3
        print("Number of data after standardizing:", self.df.shape[0])
        
        
