import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem

class smile_curate:
    def __init__(self, data, smile_col, pchem_col, keep = 'best'):
        self.data = data
        self.smile_col = smile_col
        self.pchem_col = pchem_col
        self.keep = keep
    def smile_norm(self, data, smile_col):
        df = data.dropna(subset = smile_col)
        df['Canonical_Smiles'] = df[smile_col].apply(Chem.CanonSmiles)
        return df
    def curate(self):
        df = self.smile_norm(data=self.data, smile_col=self.smile_col)
        if self.keep == 'best':
            df = df.sort_values(by=self.pchem_col, ascending=False)
            df_dropdup = df.drop_duplicates(subset=['Canonical_Smiles'], keep="first")
        elif self.keep == 'worst':
            df = df.sort_values(by=self.pchem_col, ascending=True)
            df_dropdup = df.drop_duplicates(subset=['Canonical_Smiles'], keep="first")
           
        print(df_dropdup.shape)
        self.df = df_dropdup
