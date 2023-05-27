import numpy as np

def drop_nan(df, l):
    for i in range(l):
        if np.isnan(df['拼团编号'][i]):
            df = df.drop(i)
            print(i)

    return df
