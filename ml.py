import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm,preprocessing
import pandas as pd


FEATURES =  ['DE Ratio',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']

def Build_Data_Set():
    data_df = pd.DataFrame.from_csv("key_stats.csv")
    data_df = data_df.reindex(np.random.permutation(data_df.index))

    X = np.array(data_df[FEATURES].values)

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())

    X = preprocessing.scale(X)
    X = StandardScaler().fit_transform(X)

    return X,y

def Analysis():

    size = 2000
    X, y = Build_Data_Set()
    print(len(X))

    
    clf = svm.SVC(kernel="rbf")
    clf.fit(X[:size],y[:size])

    y_pred = clf.predict(X[size+1:])
    y_true = y[size+1:]

    print accuracy_score(y_true, y_pred)



Analysis()
