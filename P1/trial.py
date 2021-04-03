# This is a trial to analyse VSCode's performance concerning Python.

# pandas.Series(data=None , index=None , dtype=None,copy = false)
# pandas.DataFrame(data=None , index=None , dtype=None,copy = false)

import pandas as pd 
import numpy as np
arr = np.arange(0,30,3).reshape(5,2) 
df = pd.DataFrame(arr,index = None, columns=['Even', 'Odd'])
print (arr)
print(df)
