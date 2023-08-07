import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from statsmodels.stats.proportion import proportions_ztest
from sklearn.impute import KNNImputer
import miceforest as mf
from sklearn.neighbors import LocalOutlierFactor

warnings.simplefilter(action="ignore")
df = pd.read_csv("diabetes.csv")
df.columns = df.columns.str.lower()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################


def check_df(dataframe, head = 5):
    print("####SHAPE####")
    print(dataframe.shape)

    print("####TYPES####")
    print(dataframe.dtypes)

    print("####HEAD####")
    print(dataframe.head(head))

    print("####TAIL####")
    print(dataframe.tail(head))

    print("####NA####")
    print(dataframe.isnull().sum())

    print("####QUANTILES####")
    print(dataframe.quantile([0, 0.01, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

###################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################


def grab_col_names(dataframe, cat_th = 10, car_th = 30):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat =[col for col in dataframe.columns
                  if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################
def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot = True)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################
def num_summary(dataframe, col_name, quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99], plot = False):
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        sns.histplot(x = dataframe[col_name],data = dataframe)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot = True)


##################################
# NUMERİK DEĞİŞKENLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
##################################
def target_summary_with_num(dataframe, col_name, target):
    print(dataframe.groupby(target)[col_name].mean())


for col in num_cols:
    target_summary_with_num(df, col, "outcome")


##################################
# KATEGORİK DEĞİŞKENLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
##################################
def target_summary_with_cat(dataframe, col_name, target):
    print(dataframe.groupby(col_name)[target].mean())

for col in cat_cols:
    target_summary_with_cat(df, col, "outcome")

##################################
# AYKIRI GÖZLEM ANALİZİ
##################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = quartile3 + 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis = None):
        print("Column name:", col_name)
        print("Low limit:", low_limit)
        print("Up limit:", up_limit)
        print(f"{col_name} outlier count", len(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]))
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)], end = "\n\n\n")
        return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis = None)
    else:
        print(f"There are no outliers according to the threshold value given in the {col_name} column")
        return False


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit


for col in num_cols:
     check_outlier(df, col)
    #if check_outlier(df, col):
     #   replace_with_thresholds(df, col)

##################################
# EKSİK GÖZLEM ANALİZİ
##################################
df.isnull().sum()
zero_columns = [col for col in num_cols if (df[col].min() == 0 and col not in ["pregnancies"])]#0 olmaması gerekip 0 olan değerler(nan ile değiştirilecek)

def change_values_with_nan(dataframe, col, value = 0, target = np.nan):
    dataframe[col] = np.where(dataframe[col] == value, target, dataframe[col])

for col in zero_columns:
    change_values_with_nan(df, col)

def missing_values_table(dataframe, na_name = True):
    na_columns = [col for col in dataframe.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return list(missing_df.iloc[::-1].index)


na_columns = missing_values_table(df, na_name=True)

#Eksik değerlerin bağımlı değişken ile incelenmesi
def missing_vs_target(dataframe, col, target):
    temp_df = dataframe.copy()
    temp_df[col + "_na_flag"] = np.where(dataframe[col].isnull(), 0, 1)
    print(temp_df.groupby(col + "_na_flag").agg({"outcome" : ["mean", "count"]}))

    test_stat, pvalue = proportions_ztest(count=[temp_df.loc[temp_df[col + "_na_flag"] == 1, target].sum(),
                                                 temp_df.loc[temp_df[col + "_na_flag"] == 0, target].sum()],

                                          nobs=[temp_df.loc[temp_df[col + "_na_flag"] == 1, target].shape[0],
                                                temp_df.loc[temp_df[col + "_na_flag"] == 0, target].shape[0]])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

for col in na_columns:
    missing_vs_target(df, col, "outcome")

##################################
# KORELASYON ANALİZİ
##################################
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# BASE MODEL KURULUMU
##################################
base_df = df.dropna()
y = base_df["outcome"]
X = base_df.drop("outcome", axis=1).dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

#Accuracy: 0.75
#Recall: 0.688
#Precision: 0.54
#F1: 0.6
#Auc: 0.73

"""
def imputation_na_with_rf(dataframe, col):#rf ile doldurmak overfit etkisi yarattı
    temp_model = RandomForestRegressor()
    temp_df_train = dataframe.loc[dataframe[col].notnull()].dropna().copy()
    temp_df_nan = dataframe.loc[dataframe[col].isnull(), dataframe.columns != col].dropna().copy()
    x_train = temp_df_train.loc[temp_df_train[col].notnull(), temp_df_train.columns != col]
    y_train = temp_df_train.loc[temp_df_train[col].notnull(), col]
    if temp_df_nan.empty or temp_df_train.empty:
        print(f"Empty dataframe, imputation failed for column : {col}")
        return False
    else:
        temp_model.fit(x_train, y_train)
        y = temp_model.predict(temp_df_nan.loc[:, temp_df_nan.columns != col])
        print(f"Mean: {y.mean()}, Min: {y.min()}, Max: {y.max()}, Column : {col}, Len : {len(y)}")
        sns.histplot(data=y)
        plt.show()
        return temp_df_nan.index, y

length = len(na_columns)
del_df = df.copy()

for index, col in enumerate(na_columns):#önce en az eksik olan yerden başlıyoruz
    if (imputation_na_with_rf(del_df, col) == False) and (index < (length * 3) - 1):
        na_columns.append(col)
    elif type(imputation_na_with_rf(del_df, col)) != bool:
        indexes, y = imputation_na_with_rf(del_df, col)
        del_df.loc[indexes, col] =  y
"""

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER PROBLEMİ
##################################

####MICEFOREST(LightGBM) ile doldurma####
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df_imputed = df.copy()

kds = mf.ImputationKernel(
  df_imputed,
  save_all_iterations=False,
  random_state=100
)

kds.mice(10)
df_imputed = kds.complete_data()

df.isnull().sum()
df = df_imputed

"""
corr_matrix = df.drop("outcome", axis = 1).corr().abs()
np.fill_diagonal(corr_matrix.values, np.nan)



####GROUPBY ile doldurma####
def fill_na_with_cor(dataframe, na_col, corr_th = 0.30):
    temp_df = dataframe.copy()
    if corr_matrix.loc[na_col].sort_values(ascending = False)[0] < corr_th:
        print(f"Correlation matrix is empty, filling failed for column : {na_col}")
        return False
    else:
        gr_cols = ["temp_col_1", "temp_col_2"]
        temp_df.loc[:, gr_cols[0]] = pd.qcut(x =  df[corr_matrix[na_col].sort_values(ascending = False)[0:1].index.values[0]], q = 5)
        temp_df.loc[:, gr_cols[1]] = pd.qcut(x =  df[corr_matrix[na_col].sort_values(ascending = False)[1:2].index.values[0]], q = 5)
        return temp_df[na_col].fillna(temp_df.groupby(gr_cols)[na_col].transform("mean"))
    

gr_df = df.copy()
for col in na_columns:
    gr_df[col] = fill_na_with_cor(df, col)
gr_df = gr_df.dropna()

####KNNIMPUTE ile doldurma####

# değişkenlerin standartlaştırılması
scaler = RobustScaler()
dff = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
dff.head()


imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)


plt.subplot(1,2,1)
sns.histplot(gr_df.loc[df["insulin"].isnull(), "insulin"])
plt.subplot(1,2,2)
sns.histplot(dff.loc[df["insulin"].isnull(), "insulin"])
plt.show()



####Groupby VS KnnImputer VS MiceForest####
print("GROUPBY")
X = gr_df.drop("outcome", axis = 1)
y = gr_df["outcome"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

y_pred = RandomForestClassifier().fit(X_train, y_train).predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

print("IMPUTER")

X = dff.drop("outcome", axis = 1)
y = dff["outcome"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

y_pred = RandomForestClassifier().fit(X_train, y_train).predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

print("MICEFOREST")

X = df_imputed.drop("outcome", axis = 1)
y = df_imputed["outcome"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

y_pred = RandomForestClassifier().fit(X_train, y_train).predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")
"""


##################################
# AYKIRI DEĞER PROBLEMİ
##################################
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)#sütunlar bireysel olarak üst veya alt sınıra göre değiştirildi


##################################
# FEATURE EXTRACTION
##################################
# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["age"] >= 21) & (df["age"] < 50), "new_age_cat"] = "mature"
df.loc[(df["age"] >= 50), "new_age_cat"] = "senior"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df["bmi_cat"] = pd.cut(x=df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["underweight", "healthy", "overweight", "obese"])

#glukoz değerini kategorikleştirme
df["glucose_cat"] = pd.cut(x=df['glucose'], bins=[0, 70, 100, 125, df["glucose"].max()],
                       labels=["low", "normal", "hidden", "high"])
# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma
def cat_col_concat(col1, col2):
    return col1+col2

df["age_bmi_cat"] = df[["bmi_cat", "new_age_cat"]].apply(lambda x : cat_col_concat(x["bmi_cat"],x["new_age_cat"]), axis = 1)

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df["age_glucose_cat"] = df[["new_age_cat", "glucose_cat"]].apply(lambda x : cat_col_concat(x["new_age_cat"],x["glucose_cat"]), axis = 1)

#insulin değişkenini kategorikleştirmek
df["insulin_score"] =df["insulin"].apply(lambda x : "normal" if 16 <= x <= 166 else "abnormal")

#glukoz ve insülin değerlerinin çarpılması
df["glucose*insulin"] = df["glucose"] * df["insulin"]

#glukoz ile hamilelik sayısının çarpılması
df["glucose*pregnancies"] = df["glucose"] * df["pregnancies"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################
num_cols.append("outcome")
lof_df = df.copy()
scaler = RobustScaler()
lof_df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(lof_df[num_cols])

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:20]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')#7. değerden sonra grafik yataylaşıyor
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th][num_cols]

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df = df.drop(axis=0, labels = df[df_scores < th].index)#aykırılık gösteren 15 gözlem veri setinden kaldırıldı.

#############################################
# Rare Encoding
#############################################
df["bmi_cat"].value_counts() / len(df) <= 0.01

def rare_analyser(dataframe, col_name, target = "outcome", rare_perc = 0.05):
    if (dataframe[col_name].value_counts() / len(df) <= rare_perc).any(axis = None):
        print(f"Rare value detected for column {col_name}", end = "\n\n")
        print(dataframe.groupby(col_name).agg({target : ["mean", "count"]}).sort_values([("outcome","count")], ascending = False), end = "\n\n\n")
        return True
    else:
        return False


def rare_encoding(dataframe, col_name, target = "outcome", rare_perc = 0.05):
    """
    Rare değer içeren kategorik sütunların rare değerlerine onlara en yakın çıktı ortalamasına ait değeri atar.
    Parameters
    ----------
    dataframe : Dataframe
    col_name : str
        Rare analizi ve ataması yapılacak kategorik kolon
    target : str
        Dataframeye ait çıktı sütun ismi
    rare_perc : float
        rare_perc * len(dataframe) eğer bu değerden az sayıda kategorik değişken değeri varsa kolonda rare değişken vardır demektir.
    Returns
    -------
        dataframe : Dataframe
    Examples
    --------
            output
               mean count
    cat_col
    a       0.70   4680
    b       0.35   1890
    c       0.11   2101
    d       0.62     74
    Bu dataframede d değeri veri setine göre sayısı az olan bir kategorik değişkendir ve encode edildiği zaman d sütununun çoğu
    0 değerinden oluşacaktır. Bu değerin yerine ona ortalama olarak en yakın değer olan a değerini atanacak.
    """
    if rare_analyser(dataframe, col_name):
        th = len(dataframe) * rare_perc
        temp_df = dataframe.groupby(col_name).agg({target: ["mean", "count"]})
        len_of_rare = len(temp_df[temp_df[(target, "count")] < th])
        for i in range(len_of_rare):
            dct = {}
            cat_count_index = temp_df[temp_df[(target, "count")] < th].index.values[i]
            cat_mean_value = temp_df.loc[cat_count_index, [(target, "mean")]].values[0]
            imputed_value = temp_df.iloc[(temp_df[(target, "mean")] - cat_mean_value).abs().argsort()[1]].name
            if imputed_value == cat_count_index:
                imputed_value = temp_df.iloc[(temp_df[(target, "mean")] - cat_mean_value).abs().argsort()[0]].name
            dct[cat_count_index] = imputed_value
        dataframe[col_name] = dataframe[col_name].apply(lambda x : dct[x] if x in dct.keys() else x)
    return dataframe[col_name]
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.remove("outcome")
df

for col in cat_cols:
    rare_analyser(df, col)

for col in cat_cols:
    while rare_analyser(df, col):
        df[col] = rare_encoding(df, col)

##################################
# ENCODING
##################################
for col in cat_cols:
    print("####COLUMN NAME : ", col ,"####")
    print("Unique values : ", df[col].unique())
    maplist = [[None, None]] *  df[col].nunique()
    for index, unique_value in enumerate(df[col].unique()):
        print("Name of unique value : ", unique_value)
        maplist[index] = [unique_value, int(input())]
    print(maplist)
    enc = OrdinalEncoder()
    enc.fit(maplist)
    df[col+"_ord"] = enc.fit_transform(df[[col]])



cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.remove("outcome")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first = True)



#############################################
# SCALING
#############################################
scaler = StandardScaler()
cat_cols_ord = [col + "_ord"for col in cat_cols]
df[num_cols + cat_cols_ord] = scaler.fit_transform(df[num_cols + cat_cols_ord])

#############################################
# MODELLEME
#############################################
y = df["outcome"]
X = df.drop("outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

#BASE MODEL
#Accuracy: 0.75
#Recall: 0.688
#Precision: 0.54
#F1: 0.6
#Auc: 0.73

#MODEL 1
#Accuracy: 0.77
#Recall: 0.646
#Precision: 0.69
#F1: 0.67
#Auc: 0.74

#MODEL 2 (WITHOUT USING MICEFOREST)
#Accuracy: 0.78
#Recall: 0.703
#Precision: 0.64
#F1: 0.67
#Auc: 0.76

#MODEL 3 (USING ORDINAL ENCODER)
#Accuracy: 0.79
#Recall: 0.672
#Precision: 0.61
#F1: 0.64
#Auc: 0.75


##################################
# FEATURE IMPORTANCE
##################################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)



#modelleme aşamasında olasılığı yüksek değerlerin kaçı doğru ona bak
lst = rf_model.predict_proba(X_test)
test_df = pd.DataFrame({"0" : [val[0] for val in lst], "1" : [val[1] for val in lst]})

indexes = test_df[(test_df["0"] > 0.85) | (test_df["1"] > 0.85)].index

y_pred = rf_model.predict(X_test.iloc[indexes])

print(f"Accuracy: {round(accuracy_score(y_pred, y_test.iloc[indexes]), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test.iloc[indexes]), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test.iloc[indexes]), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test.iloc[indexes]), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test.iloc[indexes]), 2)}")
