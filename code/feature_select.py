from collections import Counter
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd
from scipy.stats import ranksums, ttest_ind, levene, kstest
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest

data0_filePath = "./path/to/data0.csv"
data1_filePath = "./path/to/data1.csv"
info_filePath = "./path/to/info.xlsx"
data0 = pd.read_csv(data0_filePath)
data1 = pd.read_csv(data1_filePath)
info = pd.read_excel(info_filePath)

rows0, cols0 = data0.shape
rows1, cols1 = data1.shape
data0.insert(0,'label',[0] * rows0)
data1.insert(0,'label',[1] * rows1)
data = pd.concat([data0,data1])
data = shuffle(data)
data = data.fillna(0)
data.index = range(len(data))

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(test_size=0.3)
for train_index, test_index in split.split(data, data['label']):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]


# clinical features
trains_AD = train_set['AD']
info_train = info.loc[info['AD'].isin(trains_AD)]

data0 = info_train[info_train['label'] == 0]
data1 = info_train[info_train['label'] == 1]

for colName in info.columns[2:]:
    if colName == 'Sex':
        data0_col = Counter(data0[colName])
        data1_col = Counter(data1[colName])

        a = data0_col[0]
        b = data1_col[0]
        c = data0_col[1]
        d = data1_col[1]
        cross_table = np.array([[a, b],
                                [c, d]])
        n = a + b + c + d
        Ta = ((a + b) * (a + c)) / n
        Tb = ((a + b) * (b + d)) / n
        Tc = ((a + c) * (c + d)) / n
        Td = ((c + d) * (b + d)) / n
        T = min(Ta, Tb, Tc, Td)
        if ((T >= 5) & (n >= 40)):
            kf = chi2_contingency(cross_table)
        elif ((T < 5) & (T >= 1) & (n >= 40)):
            kf = chi2_contingency(cross_table, correction=True)
        else:
            kf = fisher_exact(cross_table, alternative='greater')
        print(colName, "%.4f" % kf[1])
    elif colName == 'TNM':
        data0_col = Counter(data0[colName])
        data1_col = Counter(data1[colName])
        #         print(colName, ':\n\t data0_col: ', data0_col, '\t data1_col: ', data1_col)
        a = data0_col[2]
        b = data1_col[2]
        c = data0_col[3]
        d = data1_col[3]
        e = data0_col[4]
        f = data1_col[4]
        cross_table = np.array([[a, b],
                                [c, d],
                                [e, f]])

        n = a + b + c + d + e + f
        Ta = ((a + b) * (a + c + e)) / n
        Tb = ((a + b) * (b + d + f)) / n
        Tc = ((a + c + e) * (c + d)) / n
        Td = ((c + d) * (b + d + f)) / n
        Te = ((a + c + e) * (e + f)) / n
        Tf = ((e + f) * (b + d + f)) / n
        T = min(Ta, Tb, Tc, Td, Te, Tf)

        if ((T >= 5) & (n >= 40)):
            kf = chi2_contingency(cross_table)
        elif ((T < 5) & (T >= 1) & (n >= 40)):
            kf = chi2_contingency(cross_table, correction=True)
        else:
            kf = fisher_exact(cross_table, alternative='greater')
        print(colName, "%.4f" % kf[1])
    elif colName == 'T':
        data0_col = Counter(data0[colName])
        data1_col = Counter(data1[colName])
        #         print(colName, ':\n\t data0_col: ', data0_col, '\t data1_col: ', data1_col)
        a = data0_col['T1-2']
        b = data1_col['T1-2']
        c = data0_col['T3']
        d = data1_col['T3']
        e = data0_col['T4']
        f = data1_col['T4']
        cross_table = np.array([[a, b],
                                [c, d],
                                [e, f]])

        n = a + b + c + d + e + f
        Ta = ((a + b) * (a + c + e)) / n
        Tb = ((a + b) * (b + d + f)) / n
        Tc = ((a + c + e) * (c + d)) / n
        Td = ((c + d) * (b + d + f)) / n
        Te = ((a + c + e) * (e + f)) / n
        Tf = ((e + f) * (b + d + f)) / n
        T = min(Ta, Tb, Tc, Td, Te, Tf)

        if ((T >= 5) & (n >= 40)):
            kf = chi2_contingency(cross_table)
        elif ((T < 5) & (T >= 1) & (n >= 40)):
            kf = chi2_contingency(cross_table, correction=True)
        else:
            kf = fisher_exact(cross_table, alternative='greater')
        print(colName, "%.4f" % kf[1])
    elif colName == 'N':
        data0_col = Counter(data0[colName])
        data1_col = Counter(data1[colName])
        #         print(colName, ':\n\t data0_col: ', data0_col, '\t data1_col: ', data1_col)
        a = data0_col['N0']
        b = data1_col['N0']
        c = data0_col['N1']
        d = data1_col['N1']
        e = data0_col['N2-3']
        f = data1_col['N2-3']
        cross_table = np.array([[a, b],
                                [c, d],
                                [e, f]])

        n = a + b + c + d + e + f
        Ta = ((a + b) * (a + c + e)) / n
        Tb = ((a + b) * (b + d + f)) / n
        Tc = ((a + c + e) * (c + d)) / n
        Td = ((c + d) * (b + d + f)) / n
        Te = ((a + c + e) * (e + f)) / n
        Tf = ((e + f) * (b + d + f)) / n
        T = min(Ta, Tb, Tc, Td, Te, Tf)

        if ((T >= 5) & (n >= 40)):
            kf = chi2_contingency(cross_table)
        elif ((T < 5) & (T >= 1) & (n >= 40)):
            kf = chi2_contingency(cross_table, correction=True)
        else:
            kf = fisher_exact(cross_table, alternative='greater')
        print(colName, "%.4f" % kf[1])
    elif colName == 'M':
        continue
    else:
        if kstest(data0[colName], 'norm')[1] > 0.05 and kstest(data1[colName], 'norm')[1] > 0.05:
            if levene(data0[colName], data1[colName])[1] > 0.05:
                print(colName, ':\n\tP-value = %.2f' % ttest_ind(data0[colName], data1[colName])[1])
            else:
                print(colName, ':\n\tP-value = %.2f' % ttest_ind(data0[colName], data1[colName], equal_var=False)[1])
        else:
            print(colName, ':\n\tP-value = %.2f' % ranksums(data0[colName], data1[colName])[1])


# radiomics features
data0_train = train_set[train_set['label']==0]
data1_train = train_set[train_set['label']==1]
colnames_ttest = []

for colName in train_set.columns[2:]:
    if kstest(data0_train[colName], 'norm')[1] > 0.05 and kstest(data1_train[colName], 'norm')[1] > 0.05:
        if levene(data0_train[colName], data1_train[colName])[1] > 0.05:
            if ttest_ind(data0_train[colName], data1_train[colName])[1] < 0.05:
                colnames_ttest.append(colName)
        else:
            if ttest_ind(data0_train[colName], data1_train[colName], equal_var=False)[1] < 0.05:
                colnames_ttest.append(colName)
    else:
        if ranksums(data0_train[colName], data1_train[colName])[1] < 0.05:
            colnames_ttest.append(colName)

if 'AD' not in colnames_ttest: colnames_ttest = ['AD'] + colnames_ttest
if 'label' not in colnames_ttest: colnames_ttest = ['label'] + colnames_ttest

train_set = train_set[colnames_ttest]

# lasso for further feature selection
X = train_set[train_set.columns[2:]]
y = train_set['label']
X = X.apply(pd.to_numeric,errors = 'ignore')
colNames = X.columns
X = X.fillna(0)
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

alphas = np.logspace(-3,1,50)
model_lassoCV_gtv = LassoCV(alphas = alphas, cv = 10, max_iter = 100000).fit(X,y)
coefs_gtv = model_lassoCV_gtv.path(X,y,alphas = alphas, max_iter = 100000)[1].T
coef = pd.Series(model_lassoCV_gtv.coef_,index = X.columns)
colnames_lasso = coef[coef != 0].index
colnames_lasso = colnames_lasso.tolist()

if 'AD' not in colnames_lasso:colnames_lasso = ['AD']+colnames_lasso
if 'label' not in colnames_lasso:colnames_lasso = ['label']+colnames_lasso
train_set = train_set[colnames_lasso]

# MI for further feature selection
X = train_set[train_set.columns[2:]]
y = train_set['label']
X = X.apply(pd.to_numeric,errors = 'ignore')
colNames = X.columns
X = X.fillna(0)
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
print(X.shape)

X = pd.DataFrame(X, columns=X.columns)
X = X.apply(pd.cut, bins=200, labels=False)


result_gtv = MIC(X,y)
k = result_gtv.shape[0] - sum(result_gtv <= 0)
print(k)

selector_gtv = SelectKBest(MIC, k=4)
x_MI_gtv = selector_gtv.fit_transform(X, y)
x_MI_gtv = pd.DataFrame(x_MI_gtv,columns = X.columns[selector_gtv.get_support(indices=True)])

names = x_MI_gtv.keys().tolist()
if 'AD' not in names:names = ['AD']+names
if 'label' not in names:names = ['label']+names

train_set = train_set[names]