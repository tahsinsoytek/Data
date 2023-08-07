import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor
from catboost import CatBoostClassifier

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("Customer-Churn.csv")
df.columns = df.columns.str.lower()


##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


###################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################
def grab_col_names(dataframe, cat_th = 10, car_th = 30):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

val = df[df["totalcharges"].index == 936]["totalcharges"].values[0]#totalcharges sütununda boş değer var ama metin olarak
df.loc[val == df["totalcharges"], "totalcharges"] = np.nan
df["totalcharges"] = df["totalcharges"].astype(float)


cat_cols, num_cols, cat_but_car = grab_col_names(df)#totalcharges değişkeni object tipinde olduğundan dolayı kardinal değişken olarak gözüküyor.


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################
def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x = dataframe[col_name])
        plt.show()

for col in cat_cols:
    cat_summary(df, col, True)

##################################
# SAYISAL DEĞİŞKENLERİN ANALİZİ
##################################,
def num_summary(dataframe, col_name, quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99], plot = False):
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        sns.histplot(x=dataframe[col_name], data=dataframe)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

df["tenure"].value_counts()
#tenure değişkeninde 1 aylık 612 tane değer var hatta 2000'e yakın kişi 0-10 arasında.
#Ama 1'den sonra gelen değer 72(362 adet) sonra 2, 3, 4 diye devam ediyor ardından 71 geliyor.


df[df["contract"] == "Month-to-month"]["tenure"].hist(bins=20)#Aydan aya kontrat yapanların kaldığı ay sayısı küçük sayılardan oluşmakta
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()


df[df["contract"] == "Two year"]["tenure"].hist(bins=20)#İki yıllık kontrat yapanların kaldığı ay sayısı büyük sayılardan oluşmakta
plt.xlabel("tenure")
plt.title("Two year")
plt.show()



##################################
# NUMERİK DEĞİŞKENLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
##################################
def target_summary_with_num(dataframe, col_name, target):
    print(dataframe.groupby(target)[col_name].mean())

for col in num_cols:
    target_summary_with_num(df, col, "churn")

#Tenure : Terk edenlerin  değeri bariz şekilde düşük
#MonthlyCharges : Terk edenlerden tahsil edilen tutar biraz fazla
#TotalCharges : Terk edenlerden tahsil edilen tutar az
##TotalCharges ile MonthlyCharges sütunları hedef değişkene göre ters davranıyor


##################################
# KATEGORİK DEĞİŞKENLERİN HEDEF DEĞİŞKENE GÖRE ANALİZİ
##################################
def target_summary_with_cat(dataframe, col_name, target):
    print(col_name)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col_name)[target].mean(),
                        "Count": dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}), end="\n\n\n")


df["churn"] = df["churn"].apply(lambda x :0 if x == "No" else 1)

for col in cat_cols:
    target_summary_with_cat(df, col, "churn")

##################################
# AYKIRI GÖZLEM ANALİZİ
##################################
def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = quartile3 + 1.5 * interquantile_range
    return low_limit, up_limit

def check_outliers(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        print("Column name:", col_name)
        print("Low limit:", low_limit)
        print("Up limit:", up_limit)
        print(f"{col_name} outlier count",
              len(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]))
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)], end="\n\n\n")
        return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)
    else:
        return False

def grab_outliers(dataframe, col_name, q1 = 0.05, q3 = 0.95, index = False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    print(df[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)])
    if index:
        return df[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].index

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    return dataframe[col_name]

for col in num_cols:
    check_outliers(df, col)#default quantile değerlerine göre aykırı değer gözükmüyor


##################################
# KORELASYON ANALİZİ
##################################
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Tenure ile TotalCharges arasında pozitif çok güçlü korelasyon var
#MonthlyCharges ile TotalCharges arasında güçlü pozitif korelasyon var


##################################
# BASE MODEL KURULUMU
##################################
base_df = df.dropna()
y = base_df["churn"]
X = base_df.drop(["churn", "customerid"], axis=1).dropna()
X = pd.get_dummies(X, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

#Accuracy: 0.79
#Recall: 0.622
#Precision: 0.49
#F1: 0.55
#Auc: 0.73

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER PROBLEMİ
##################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df[na_columns] = df.groupby("new_tenure_year")[na_columns].transform(lambda x : x.fillna(x.median()))#boş değerlerin kırılıma göre doldurulması

df.isnull().sum()
##################################
# FEATURE EXTRACTION
##################################
# Tenure değişkeninden yıllık kategorik sütun türetme
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "new_tenure_year"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "new_tenure_year"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "new_tenure_year"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "new_tenure_year"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "new_tenure_year"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "new_tenure_year"] = "5-6 Year"

# Cinsiyet ve yaş durumuna göre yeni kategorik değişken oluşturma
df.loc[(df["gender"] == "Female") & (df["seniorcitizen"] == 0), "gender_senior_citizen"] = "YoungFemale"
df.loc[(df["gender"] == "Female") & (df["seniorcitizen"] == 1), "gender_senior_citizen"] = "SeniorFemale"
df.loc[(df["gender"] == "Male") & (df["seniorcitizen"] == 0), "gender_senior_citizen"] = "YoungMale"
df.loc[(df["gender"] == "Male") & (df["seniorcitizen"] == 1), "gender_senior_citizen"] = "SeniorMale"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["new_engaged"] = df["contract"].apply(lambda x : 1 if x != "Month-to-month" else  0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["new_no_prot"] = df.apply(lambda x : 1 if (x["onlinebackup"] == "Yes") or (x["deviceprotection"] == "Yes") or (x["techsupport"] == "Yes") else 0, axis = 1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["new_young_not_engaged"] = df.apply(lambda x : 1 if (x["seniorcitizen"] == 0) and (x["contract"] == "Month-to-month") else 0, axis = 1)

# Kişinin toplam aldığı servis sayısı
lst = ['phoneservice', 'internetservice', 'onlinesecurity',
                               'onlinebackup', 'deviceprotection', 'techsupport',
                               'streamingtv', 'streamingmovies']
df["new_totalservices"] = (df[lst] == 'Yes').sum(axis = 1)

# Herhangi bir streaming hizmeti alan kişiler
df["new_flag_any_streaming"] = df.apply(lambda x : 1 if (x["streamingtv"] == "Yes") or  (x["streamingmovies"] == "Yes") else 0, axis = 1)

# Kişi otomatik ödeme yapıyor mu?
df["new_flag_autopayment"] = df["paymentmethod"].apply(lambda x : 1 if "automatic" in x else 0)

# Ortalama aylık ödeme
df["new_avg_charges"] = df["totalcharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["new_increase"] = df["new_avg_charges"] / (df["monthlycharges"] + 1)

# Servis başına ücret
df["new_avg_service_fee"] = df["monthlycharges"] / (df["new_totalservices"] + 1)


##################################
# ENCODING
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.append("new_totalservices")
cat_cols.remove("new_totalservices")

nominal_cols = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'onlinesecurity', 'onlinebackup',
'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'paperlessbilling', 'paymentmethod', 'gender_senior_citizen',
'seniorcitizen', 'new_engaged', 'new_no_prot', 'new_young_not_engaged', 'new_flag_any_streaming', 'new_flag_autopayment']

ordinal_cols = ['internetservice', 'contract', 'new_tenure_year']

#########################################
#ORDİNAL VERİLERİN ENCODE EDİLMESİ
#########################################
for col in ordinal_cols:
    print("####COLUMN NAME : ", col ,"####")
    print("Unique values : ", df[col].unique())
    maplist = [[None, None]] *  df[col].nunique()
    for index, unique_value in enumerate(df[col].unique()):
        print("Name of unique value : ", unique_value)
        maplist[index] = [unique_value, int(input())]
    print(maplist)
    enc = OrdinalEncoder()
    enc.fit(maplist)
    df[col] = enc.fit_transform(df[[col]])

#########################################
#NOMİNAL VERİLERİN ENCODE EDİLMESİ
#########################################
cat_cols.remove("churn")
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe

df = one_hot_encoder(df, nominal_cols, drop_first = True)



#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################
num_cols.append("churn")#çıktı ile beraber analiz etmek için eklendi
lof_df = df.copy()
scaler = MinMaxScaler((-1,1))
lof_df[num_cols] = scaler.fit_transform(df[num_cols])
lof_df[num_cols].head()
lof_df.shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(lof_df[num_cols])

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:20]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 1000], style='.-')#100 civarından sonra grafik yataylaşıyor
plt.show()

th = np.sort(df_scores)[100]

df[df_scores < th][num_cols]

df[num_cols].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df = df.drop(axis=0, labels = df[df_scores < th].index)#aykırılık gösteren 100 gözlem veri setinden kaldırıldı.

num_cols.remove("churn")


#############################################
# SCALING
#############################################
scaler = StandardScaler()
df[num_cols + ordinal_cols] = scaler.fit_transform(df[num_cols + ordinal_cols])


#############################################
# MODELLEME
#############################################


########################
#RANDOM FOREST
########################
y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


########################
#CATBOOST
########################
catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")



#BASE MODEL RF
#Accuracy: 0.79
#Recall: 0.622
#Precision: 0.49
#F1: 0.55
#Auc: 0.73

#LAST MODEL RF
#Accuracy: 0.79
#Recall: 0.638
#Precision: 0.48
#F1: 0.55
#Auc: 0.73


#LAST MODEL CATBOOST
#Accuracy: 0.81
#Recall: 0.689
#Precision: 0.52
#F1: 0.59
#Auc: 0.77


def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(25, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')
plot_feature_importance(rf_model.feature_importances_, X.columns, 'Random Forest')