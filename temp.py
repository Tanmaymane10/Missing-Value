import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

ds = pd.read_csv('rawproperties.csv')

print(ds.isnull().sum())
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
ds_imputed = pd.DataFrame(imputer.fit_transform(ds), columns=ds.columns)
ds_imputed.to_csv('rawproperties.csv', index=False)




