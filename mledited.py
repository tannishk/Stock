import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm,preprocessing
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import time


##import statistics
##import numpy as np

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))



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
    ##print data_df
    X = np.array(data_df[FEATURES].values)

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())

    X = preprocessing.scale(X)
    X = StandardScaler().fit_transform(X)
    Z0 = np.array(data_df["stock_p_hancge"])
    Z1 = np.array(data_df["sp500_p_change"])
    return X,y,Z0,Z1

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

def mykernel(X, Y,gamma=None):
   
   ## X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1/X.shape[1]
    dis = euclidean_dist_matrix(X,Y)
    # exponentiate K in-place
    return np.exp((gamma* dis)) + np.dot(X,Y.T)

size = 2094
invest_amount = 10000
total_invests = 0
if_market = 0
if_strat = 0
X, y , Z0,Z1= Build_Data_Set()
print(len(X))
test_size = len(X) - size -1 
    
start = time.clock()
clf = svm.SVC(kernel=mykernel)
clf.fit(X[:size],y[:size])

y_pred = clf.predict(X[size+1:])
y_true = y[size+1:]
time_taken = time.clock()-start
print time_taken,"Seconds"

for x in range(1, test_size+1):
    if y_pred[-x] == 1:
        invest_return = invest_amount + (invest_amount * (Z0[-x]/100))
        market_return = invest_amount + (invest_amount * (Z1[-x]/100))
        total_invests += 1
        if_market += market_return
        if_strat += invest_return

print accuracy_score(y_true, y_pred)

print precision_recall_fscore_support(y_true, y_pred, average='macro')

print "Total Trades:", total_invests
print "Ending with Strategy:",if_strat
print "Ending with Market:",if_market

compared = ((if_strat - if_market) / if_market) * 100.0
do_nothing = total_invests * invest_amount

avg_market = ((if_market - do_nothing) / do_nothing) * 100.0
avg_strat = ((if_strat - do_nothing) / do_nothing) * 100.0


    
print "Compared to market, we earn",str(compared)+"% more" 
print "Average investment return:", str(avg_strat)+"%" 
print "Average market return:", str(avg_market)+"%" 



