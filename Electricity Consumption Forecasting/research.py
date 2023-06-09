import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor



def loading():
    df = pd.read_csv("train.csv")
    medtimes = pd.read_csv("med.csv")
    df = preprocessing(df, "tarih")
    return df,medtimes

def preprocessing(df, col):
    df["Tarih"] = pd.to_datetime(df["Tarih"], format="%Y-%m-%d")
    df.columns = df.columns.str.lower()
    df["saat"] = df[col].dt.hour
    df["gün"] = df[col].dt.day
    df["ay"] = df[col].dt.month
    df["yıl"] = df[col].dt.year
    return df

df,medtimes = loading()

def temperature():
    x = np.empty((0))
    for i in range(1):
        print("I VALUE : ",i+1)
        print("firstT")
        firstT = float(input())

        print("lastT")
        lastT = float(input())

        print("numberofdays")
        numberofdays = float(input())

        x = np.append(x, np.arange(firstT, lastT, (lastT - firstT) / numberofdays))
    return x

#Exploratory Data Analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)

#target_vs_cat
for col in cat_cols:
    target_summary_with_cat(df, target, col)
#target_vs_num
correlation_matrix(df, num_col_target)
z = temperature()

t = np.empty((0))
for i in range(len(z)):
    t = np.append(t, np.repeat(z[i], 24))


####Feature Engineering####

def local_outlier(df):

    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df)

    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    # df_scores = -df_scores
    np.sort(df_scores)[0:5]
    df["scores"] = pd.DataFrame(df_scores)
    #scores = pd.DataFrame(np.sort(df_scores))
    df["scores"].plot(stacked=True, xlim=[0, 50], style='.-')
    plt.show()
    df["scores"][0:20]
    th = np.sort(df_scores)[12]

    df[df_scores < th]

    df[df_scores < th].shape

    df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

    df[df_scores < th].index
    df[df["scores"] < th]
    df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

local_outlier(df)


def cyclic_data_prep(df):
    for col in cols:
        df[col + "_x"] = np.sin(2. * np.pi * df[col] / float(df[col].nunique()))
        df[col + "_y"] = np.cos(2. * np.pi * df[col] / float(df[col].nunique()))
    return df


df = cyclic_data_prep(df)

for col in cols:
    plt.plot(df[col + "_x"], df[col + "_y"])
    plt.show()


def feature_clustering(df):
    for col in cols:
        kmeans = KMeans(n_clusters = best_param_for_clustering(df, col))
        kmeans.fit(df[[col + "_x", col + "_y", "dağıtılan enerji (mwh)_scaled"]])
        df[col+"label"] = kmeans.labels_

label_list = ["saatlabel_","günlabel_","aylabel_"]

for x in label_list:
    for i in range(df[x].nunique()):
        print("i value:",i+1,"col:", x)
        print(df[df[x] == i][x.replace("label_","")].unique())

saat = []
gun = []
ay = []

for row in range(list(df[df["dağıtılan enerji (mwh)"] == 0].index)[0], list(df[df["dağıtılan enerji (mwh)"] == 0].index)[-1] + 1):
    saat.append(df[df.iloc[row]["saat"] == df["saat"]]["saatlabel_"].unique()[0])
    gun.append(df[df.iloc[row]["gün"] == df["gün"]]["günlabel_"].unique()[0])
    ay.append(df[df.iloc[row]["ay"] == df["ay"]]["aylabel_"].unique()[0])

df.loc[df["dağıtılan enerji (mwh)"] == 0, "saatlabel_"] = saat
df.loc[df["dağıtılan enerji (mwh)"] == 0, "günlabel_"] = gun
df.loc[df["dağıtılan enerji (mwh)"] == 0, "aylabel_"] = ay


for col in label_list:
    df.loc[df["dağıtılan enerji (mwh)"] == 0, col].apply(label_assignment)
kmeans= KMeans()
def best_param_for_clustering(df, col):
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(df[[col + "_x", col + "_y", "dağıtılan enerji (mwh)_scaled"]])
    elbow.show()
    return elbow.elbow_value_

feature_clustering(df)

def singularization(df):
    for col in cols:
        for i in range(df[col+"label"].nunique()):
            indexes = (df[df[col + "label"] == i][col].value_counts() / df[df[col + "label"] == i][col].value_counts().sum() > 0.05).index.values
            df.loc[df[col].isin(indexes), col+"label_"] = i

singularization(df)


for col in cols:
    df = df.drop([col+"_x", col+"_y"], axis = 1)

df = df.drop(["saatlabel","günlabel","aylabel"],axis =1)
cln = pd.read_csv("Turkish calendar.csv", sep=(";")).iloc[884:2557],

cln = cln.iloc[853:2557]


def special_days():
    cln = pd.read_csv("Turkish calendar.csv", sep=(";"))
    cln = cln.iloc[853:2557].copy()
    medtimes = pd.read_csv("med.csv")
    cln.rename(columns={'DAY_OF_WEEK_SK': 'Haftanın Günü'}, inplace=True)
    cln.rename(columns={'WEEKEND_FLAG': 'Haftasonu - Haftaiçi'}, inplace=True)
    cln.rename(columns={'RAMADAN_FLAG': 'Ramazan'}, inplace=True)
    cln.rename(columns={'RELIGIOUS_DAY_FLAG_SK': 'Dini Gün'}, inplace=True)
    cln.rename(columns={'NATIONAL_DAY_FLAG_SK': 'Ulusal Gün'}, inplace=True)
    cln.rename(columns={'PUBLIC_HOLIDAY_FLAG': 'Resmi tatil'}, inplace=True)
    cln = cln.iloc[::-1]

    cln = cln[["CALENDAR_DATE", "Haftanın Günü",'Haftasonu - Haftaiçi','Ramazan' , 'Dini Gün', 'Ulusal Gün', 'Resmi tatil']]
    temp = pd.DataFrame(index = range(len(cln) * 24), columns = [["Haftanın Günü",'Haftasonu - Haftaiçi','Ramazan' , 'Dini Gün', 'Ulusal Gün', 'Resmi tatil']])
    print(cln)
    for col in cln.columns:
        t = np.empty((0))
        for i in range(len(cln)):
            t = np.append(t, np.repeat(cln.iloc[i][col], 24))
        temp[col] = t
    df[["Haftanın Günü",'Haftasonu - Haftaiçi','Ramazan' , 'Dini Gün', 'Ulusal Gün', 'Resmi tatil',"CALENDAR_DATE"]] = temp

    df["is_med"] = df["date"].astype("str").isin(medtimes["Tarih"].astype("str"))#eşleşen tarih varsa True basar değilse False basar
    df["is_med"] = np.where(df["is_med"], 1, 0)#True 1 oldu False 0
    df["dini gün"] = df["Dini Gün"].apply(lambda row: 0 if row == 100 else 1)#Dini Günler 1 diğerleri 0
    df["ulusal gün"] = df["Ulusal Gün"].apply(lambda row: 0 if row == 200 else 1)#Ulusal Günler 1 diğerleri 0
    df["resmi tatil"] = df["Resmi tatil"].apply(lambda row : 1 if row in "Y" else 0)#Resmi Tatil 1 diğerleri 0
    df["haftasonu haftaiçi"] = df["Haftasonu - Haftaiçi"].apply(lambda row : 1 if row in "Y" else 0)#Haftasonu 1 hafta içi 0
    df["ramazan"] = df["Ramazan"].apply(lambda row : 1 if row in "Y" else 0)#Ramazan 1 diğerleri 0

special_days()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

remove_list = ["dağıtılan enerji (mwh)_scaled", "saat", "gün", "ay", "Haftasonu - Haftaiçi", "Ramazan", "Dini Gün", "Ulusal Gün", "Resmi tatil", "CALENDAR_DATE", "date"]
df = df.drop(remove_list,axis = 1)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def encoding(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = encoding(df, cat_cols)


low, up = outlier_thresholds(df, "dağıtılan enerji (mwh)")
df[(df["dağıtılan enerji (mwh)"] < low) | (df["dağıtılan enerji (mwh)"] > up)][["tarih","dağıtılan enerji (mwh)"]].dt.value_counts()
check_outlier(df,"dağıtılan enerji (mwh)")


