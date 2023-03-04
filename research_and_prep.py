import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import unidecode
import gc
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)


def loading_and_merging():
    df1 = pd.read_csv("Dataset/education.csv")
    # df1.drop(["start_year_month", "end_year_month"], axis=1, inplace=True)
    df2 = pd.read_csv("Dataset/languages.csv")
    df3 = pd.read_csv("Dataset/skills.csv")
    df4 = pd.read_csv("Dataset/train_users.csv")
    df5 = pd.read_csv("Dataset/work_experiences.csv")
    df6 = pd.read_csv("Dataset/test_users.csv")
    x = pd.concat([df4, df6])

    train_user_id = df4['user_id'].unique()
    test_user_id = df6['user_id'].unique()

    x = pd.merge(x, df1, on="user_id", how="left")
    x = pd.merge(x, df2, on="user_id", how="left")
    x = pd.merge(x, df3, on="user_id", how="left")
    x = pd.merge(x, df5, on="user_id", how="left")
    return x, train_user_id, test_user_id


def load_data():
    print("Yükleniyor...")
    df = pd.read_csv("raw_data.csv")
    print("Yüklendi")
    return df


df, train_user_id, test_user_id = loading_and_merging()

df = load_data()

df.to_csv("raw_data.csv", index_label=False)

print("Bitti")

df = pd.read_csv("raw_data2.csv", low_memory=False)
print("Bitti")

df.to_csv("raw_data3.csv", index_label=False)
print("Bitti")


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


check_df(df, head=10)

df["location_y"].nunique()

df.drop(["location_y"], axis=1, inplace=True)
df["location"] = df["location_x"]
df.drop(["location_x"], axis=1, inplace=True)


df[df["location"] == "Istanbul, Istanbul, Turkey"]

df.loc[df["location"].str.contains("Istanbul")]


def missing_value_analysis_with_target(df, col, target):  # Boş değerler ile hedef değişkenin analizi
    print(col, "\n",
          df.groupby(np.where(df[col].isnull(), 0, 1)).agg({target: ["mean", "count"]}))  # 0 Null değer 1 Dolu değer




for col in df.columns:
    if df[col].isnull().any():
        missing_value_analysis_with_target(df, col, "moved_after_2019")

"""
industry 2831(2143) boş değer ve ortalaması ciddi anlamda düşük endüstri boş mu dolu mu diye atama yapılacak --> done
school_name 70 boş değer satırlar silinecek --> done
degree fazla boş değer var ve boş değerler ile anlamlı bir farklılık gözükmüyor. Kırılım veya ml yöntemi ile atama yapılacak
fields of study fazla boş değer var ve hiçbir farklılık gözükmüyor. Kırılım veya ml yöntemi ile atama yapılacak
language fazla boş değer var ve çok farklılık gözükmüyor. Kırılım veya ml yöntemi ile atama yapılacak
proficiency fazla boş değer var ve çok farklılık gözükmüyor. Kırılım veya ml yöntemi ile atama yapılacak
skill fazla boş değer var ve çok farklılık gözükmüyor. Kırılım veya ml yöntemi ile atama yapılacak
company_id boş değerlerin ortalaması 0 yani hiçbiri işten ayrılmamış company_id değerinin boş olması ciddi anlamda(bilinçli 0 olabilir)
 işten ayrılmayı etkiliyor(start_year_month)
"""

df.drop(df[df["location"].isna()].index, axis=0, inplace=True)
df.drop(df[df["school_name"].isna()].index, axis=0, inplace=True)

df["industry_na"] = np.where(df["industry"].isnull(), 1, 0)  # 1 null 0 dolu değer


def delete_na(df, col):
    df.drop(df[df[col].isna()].index, axis=0, inplace=True)


df["industry"].fillna("NaN", inplace=True)

df[["industry_na", "industry"]]
df["industry_na"].unique()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


df[df["school_name"].str.contains("Beykent University")]

cat_summary(df, "school_name")
for col in df.columns:
    cat_summary(df, col)
df.groupby("school_name").agg({"moved_after_2019": ["mean", "count"]}).sort_values(('moved_after_2019', 'count'),
                                                                                   ascending=False)

df[df["user_id"] == 16036]


def target_summary_with_cat(dataframe, target, categorical_col):
    print(dataframe.groupby(categorical_col).agg({target: ["mean", "count"]}), end="\n\n\n")


for col in df.columns:
    target_summary_with_cat(df[~df["moved_after_2019"].isna()], "moved_after_2019", col)

for col in df.columns:
    target_summary_with_cat(df[df["moved_after_2019"].notna()], "moved_after_2019", col)

dfgr = df.groupby("skill3").agg({"moved_after_2019": ["mean", "count"]}).sort_values(('moved_after_2019', 'count'),
                                                                                     ascending=False)
dfgr

dfgr = df[df["moved_after_2019"].notna()]["user_id"].unique()
len(dfgr)

len(df["user_id"].unique())

df[df["user_id"] == 32008]["skill3"].value_counts()

dfgr.columns
dfgr[dfgr[('moved_after_2019', 'mean')] > 0.575]

df[df["user_id"] == 0]["skill3"].value_counts()

df["skill3"].isna().sum()
df["skill3"] = df["skill3"].apply(lambda x: np.nan if "nan" in str(x) else x)
df.groupby("user_id")["skill3"].count()

df[df["moved_after_2019"].notna()]["skill3"].value_counts()
df.drop(df[df["na_start_year_month"] == 0].index, axis=0, inplace=True)
df[["start_year_month", "company_id", "moved_after_2019"]]

df[df["start_year_month"].isna()]
check_df(df)



df["na_start_year_month_y"] = np.where(df["start_year_month_y"].isnull(), 0, 1)

cols = df.loc[:, df.columns.str.contains("start_year_month")]

df.groupby("NA_start_year_month_y")["moved_after_2019"].mean()

df[["user_id", "start_year_month_x", "start_year_month_y"]]

df["moved_after_2019"].isnull().sum()


df = pd.read_csv("raw_data2.csv")


df.to_csv("raw_data2.csv", index_label=False)





df_train = df[["location", "user_id"]].loc[df[df["location"].str.contains("Greater")].index]

df_train[df_train["location"].str.contains("Turkey")]

df_train["location"].unique()

~df_train["location"].str.contains("Turkey")

df_train["location"].unique()
df_train["location"].value_counts()

df_train["location"] = df_train["location"].apply(
    lambda x: x.replace(x, x.replace(" ", "") + ", Turkey") if x.startswith(" ") else x)
df_train[df_train["location"].str.startswith(" ")]

df_train["location"].apply(location_lambda).value_counts()

df_train["location"].apply(location_lambda)

df_train["location"].unique()

df_train[df["location"].str.contains("Greater")].index

df_train[df_train["location"].str.contains("greater")]
df[df["location"].str.contains("Greater ")]

df_train["locationX"] = df_train["location"].apply(lambda x: x.replace("Greater ", "") if "Greater" in x else x)

df_train["location"] = df_train["location"].apply(
    lambda x: x.replace(x[0: x.index(" ") + 1], "", 1) if len(str(x).split(",")) > 2 else x)

df_train["location"][0:500] = df_train["location"].astype("str")

df_train["location"] = df_train["location"].apply(
    lambda x: x.replace("Greater", "") if x.str.contains("Greater") else x)


pd.read_csv("raw_data2.csv", nrows=1).dtypes
lst = list(pd.read_csv("raw_data2.csv", nrows=1))
tuple(lst)

df.info()

d = {i: "uint8" for i in lst}
d["user_id"] = "uint32"
d['company_count'] = "uint16"
d['moved_after_2019'] = "float64"
dict({tuple(lst): "uint8"})

del d["moved_after_2019"]
len(d.keys())

len(lst)
df.dtypes
lst = [i for i in lst if str(i) not in lst]
lst.remove("moved_after_2019")
len(lst)

df = pd.read_csv("raw_data2.csv", usecols=lst, dtype=d)
df2 = pd.read_csv("raw_data2.csv", usecols=['moved_after_2019'])
df["moved_after_2019"] = df2['moved_after_2019']
df.info()
print("done")
# df = pd.concat([df,pd.read_csv("raw_data2.csv", usecols = lst)],axis=1)

df = pd.read_csv("raw_data6.csv")

for col in lst:
    df[col] = df[col].astype("uint8")

[i for i in range(len(lst))]
print("done")
df.dtypes
df["location"] = df["location"].apply(
    lambda x: x.replace(x[0: x.index(" ") + 1], "", 1) if len(str(x).split(",")) > 2 else x)
df["location"] = df["location"].apply(lambda x: x.replace("Greater", "") if "Greater " in x else x)

del df2
gc.collect()
df["location"].nunique()
df["location"].apply()


def location_lambda(x):
    # içinde Turkey geçen sıklığı düşükleri(İzmir, İstanbul, Ankara, (Turkey) hariç) Turkey içine ata diğer kalanları yurt dışı yapılacak
    if not (("İzmir" in x) or ("Istanbul" in x) or ("Ankara" in x)) and (
            "Turkey" in x):  # bu şehirler haricindeki şehirler Turkey olarak atandı
        x = "Turkey"
    elif not ("Turkey" in x):  # diğer ülkelerdeki çalışanlar tek bir kategoriye alındı
        x = "Abroad"
    elif ("Izmir" in x):
        x = "İzmir, Turkey"
    return x



df[df["school_name"].str.contains("Boğaziçi")]["school_name"].value_counts()
df["school_name"].value_counts()[100 * df["school_name"].value_counts() / len(df) > 0.1]  # 0.5 threshold


def skill_lambda(x):
    if "\t" in str(x):
        x = str(x).replace("\t", "")
    if "*" in str(x):
        x = str(x).replace("*", "")
    return x


def proficiency_lambda(x):
    # elementary-limited_working-professional_working-full_professional-native sayısal olarak fazla olmasının anlamı var

    if "elementary" in str(x):
        return 1
    elif "limited_working" in str(x):
        return 2
    elif "professional_working" in str(x):
        return 3
    elif "full_professional" in str(x):
        return 4
    elif "native_or_bilingual" in str(x):
        return 5
    else:
        return x


def start_year_month_lambda(x):
    if x["start_year_month"].mean() == 0:
        gc.collect()
        return 0
    else:
        year = int(str(x["start_year_month"].max())[0:4]) - int(str(x["start_year_month"].min())[0:4])
        month = int(year) * 12
        month = month + int(str(x["start_year_month"].max())[4:]) - int(str(x["start_year_month"].min())[4:])
        gc.collect()
        return month


df.groupby(df["school_name"]).agg({"moved_after_2019": ["count", "mean"]}).sort_values(('moved_after_2019', 'count'),
                                                                                       ascending=False)
df["school_name"].value_counts()
df.drop("Unnamed: 0", axis=1, inplace=True)
df.to_csv("raw_data2.csv", index_label=False)
print("done")
df[df["school_name"].str.contains("Boğaziçi")]

df[df["school_name"].str.contains("Istanbul")]["school_name"].value_counts()
print("done")
df["school_name"].nunique()

df[df["school_name"].str.contains("/", na=False)]["school_name"].value_counts()
df["school_name"].value_counts()

df[df["school_name"].str.contains("Yuksek Teknoloji Enstitusu", na=False)]["school_name"].value_counts()
df[df["school_name"].str.contains("Institute of Technology", na=False)]["school_name"].value_counts()

df[df["school_name"].str.contains("Orta Dogu Technical", na=False)]

len(df["school_name"].value_counts()[100 * df["school_name"].value_counts() / len(df) > 0.1])

df["school_name"] = df["school_name"].astype(str)  # na değerler stringe çevrildi
df["degree"] = df["degree"].astype(str)
df["degree2"] = df["degree2"].astype(str)


def data_prep(df):
    # location
    df["location"] = df["location"].apply(
        lambda x: x.replace(x[0: x.index(" ") + 1], "", 1) if len(str(x).split(",")) > 2 else x)
    df["location"] = df["location"].apply(lambda x: x.replace("Greater", "") if "Greater " in x else x)
    df["location"] = df["location"].apply(
        lambda x: x.replace(x, x.replace(" ", "") + ", Turkey") if x.startswith(" ") else x)
    df["location"] = df["location"].apply(
        location_lambda)  # başında boşluk olan şehirlerin boşluğu silindi ve sonuna Turkey eklendi

    # school_name
    df["school_name"] = df["school_name"].apply(lambda x: unidecode.unidecode(x))  # ingilizce harflere çevrim
    df["school_name"] = df["school_name"].apply(lambda x: x.replace("Teknik", "Technical") if "Teknik" in x else x)
    df["school_name"] = df["school_name"].apply(lambda x: x.replace("Yuksek Teknoloji Enstitusu",
                                                                    "Institute of Technology") if "Yuksek Teknoloji Enstitusu" in x else x)
    df["school_name"] = df["school_name"].apply(
        lambda x: x.replace("Universitesi", "University") if "Universitesi" in x else x)
    df["school_name"] = df["school_name"].apply(
        lambda x: x.replace(x, "Bogazici University") if "Bosphorus University" in x else x)
    df["school_name"] = df["school_name"].apply(lambda x: x.replace(x, x.split("/")[1].strip()) if "/" in x else x)
    df["school_name"] = df["school_name"].apply(
        lambda x: x.replace(x, "Middle East Technical University") if "Orta Dogu Technical University" in x else x)
    not_rare_list = df["school_name"].value_counts()[100 * df["school_name"].value_counts() / len(df) > 0.5].index
    df["school_name"] = df["school_name"].apply(lambda x: x if x in not_rare_list else "Other")

    # degree
    df["degree2"] = df["degree"].apply(
        lambda x: x.replace(x, "Lisans Derecesi") if (("BS" in x) or ("Bachelor" in x) or ("Lisans" in x) or (
                "B.Sc." in x) or ("B.S." in x)) and (("Yüksek Lisans" not in x) and ("Önlisans" not in x) and (
                    "Ön lisans" not in x)) else x)

    df["degree2"] = df["degree2"].apply(
        lambda x: x.replace(x, "Lise Derecesi") if ("High School" in x) or ("Lise" in x) else x)
    df["degree2"] = df["degree2"].apply(
        lambda x: x.replace(x, "Ön Lisans Derecesi") if ("önlisans" in x) or ("Ön lisans" in x) or (
                    "Associate" in x) or ("Önlisans" in x) else x)

    # df["degree"] = df["degree"].apply(lambda x : x.replace(x,"Lisans Derecesi") if ("BS" in x) or ("Bachelor" in x) or ("Lisans" in x) else x).value_counts()[0:20].sum()/len(df)

    df["degree2"] = df["degree2"].apply(
        lambda x: x.replace(x, "Lisans Derecesi") if ("lisans" in x) or ("Undergraduate" in x) or (
                    "Licentiate" in x) or (
                                                             "Engineer" in x) or ("Mühendislik" in x) else x)

    df["degree2"] = df["degree2"].apply(
        lambda x: x.replace(x, "Yüksek Lisans Derecesi") if ("Graduate" in x) or ("MS" in x) or ("Master" in x) or (
                "M.S." in x) or ("MBA" in x) or ("M.Sc." in x) or ("Yüksek Lisans" in x) else x)

    df["degree2"] = df["degree2"].apply(
        lambda x: x.replace(x, "Doktora") if ("Doctor" in x) or ("PhD" in x) or ("Ph.D." in x) or ("Doktora" in x) or (
                "Dr." in x) else x)

    df["degree"] = df["degree2"].apply(
        lambda x: x.replace(x, "Other") if ("nan" not in str(x)) and ("Lisans" not in str(x)) and (
                "Doktora" not in str(x)) and ("Lise" not in str(x)) and ("Ön lisans" not in str(x)) else x)

    # language

    df["language"] = df["language"].apply(lambda x: unidecode.unidecode(str(x)))  # ingilizce harflere çevrim
    df["language"] = df["language"].apply(lambda x: x.lower())  # küçük harflere çevrim

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "English") if ("English" in str(x)) or ("englisch" in str(x)) or (
                    "ingilizce" in str(x)) or ("english" in str(x)) or ("inglizce" in str(x))
                                             or ("ingilzce" in str(x)) or ("ingilize" in str(x)) or (
                                                         "ingizice" in str(x)) or ("ingilizice" in str(x)) or (
                                                         "ingiliz" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Turkish") if ("turkce" in str(x)) or ("turkish" in str(x)) or (
                    "turkisch" in str(x)) or ("turksih" in str(x))
                                             or ("turk" in str(x)) or ("turke" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "German") if ("german" in str(x)) or ("almanca" in str(x)) or (
                "deutsch" in str(x)) or ("germany" in str(x))
                                            or ("deutsche" in str(x)) or ("germanic" in str(x)) else x)
    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Russian") if ("russian" in str(x)) or ("rusca" in str(x)) or (
                    "russkii" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Japanese") if ("japanese" in str(x)) or ("japonca" in str(x)) or (
                    "japanesse" in str(x)) or ("japanesse" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "French") if ("french" in str(x)) or ("fransizca" in str(x)) or (
                "francais" in str(x)) or ("franzosisch" in str(x))
                                            or ("fransa" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Spanish") if ("spanish" in str(x)) or ("ispanyolca" in str(x)) or (
                "espanol" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Arabic") if ("arapca" in str(x)) or ("arabic" in str(x)) or (
                "arabe" in str(x)) or ("arabish" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Italian") if ("italian" in str(x)) or ("italyanca" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Chinese") if ("chinese" in str(x)) or ("cince" in str(x)) or (
                "mandarin" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Latin") if ("latince" in str(x)) or ("latin" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Azerbaijani") if ("azerbaijani" in str(x)) or ("azerice" in str(x)) or (
                    "azeri" in str(x)) or ("azerbaycanca" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: x.replace(x, "Dutch") if ("dutch" in str(x)) or ("felemenkce" in str(x)) else x)

    df["language"] = df["language"].apply(
        lambda x: str(x).replace(str(x), "Other") if (str(x)[0].islower()) or (str(x).startswith("`")) else x)

    # industry

    lst = list(df["industry"].value_counts()[100 * df["industry"].value_counts() / len(df) > 0.78].index)

    df["industry"] = df["industry"].apply(lambda x: str(x).replace(str(x), "Other") if x not in lst else x)

    # fields_of_study
    # df[np.logical_or(df["fields_of_study"].astype(str).str.contains("Engineer"), df["fields_of_study"].astype(str).str.contains("Engineer"))]

    df["fields_of_study2"] = df["fields_of_study"].apply(
        lambda x: x.replace(x, "COMP") if "Bilgisayar Mühendisliği" in str(x) else x)
    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "COMP") if "Computer Engineering" in str(x) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "SOFT") if "Yazılım" in str(x) else x)
    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "SOFT") if "Software" in str(x) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "ELEK") if "Electrical and Electronics Engineering" in str(x) else x)
    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "ELEK") if "Elektrik ve Elektronik Mühendisliği" in str(x) else x)
    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "ELEK") if ("Electronics and Communication" in str(x)) or
                                          ("Electronic and Communication" in str(x)) else x)
    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "ELEK") if "Elektronik ve Haberleşme Mühendisliği" in str(x) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "Engineering") if ("engineer" in str(x)) or ("mühendis" in str(x))
                                                 or ("muhendis" in str(x)) or ("enginer" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "Computer Programming") if ("programla" in str(x)) or ("programm" in str(x)) or
                                                          ("programci" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: unidecode.unidecode(str(x)))  # ingilizce harflere çevrim

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "business administration") if ("business" in str(x)) or ("mba" in str(x))
                                                             or ("islet" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "IT") if ("information" in str(x)) and ("tech" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "computer science") if (("bilgisayar" in str(x)) and ("bilim" in str(x)))
                                                      or (("computer" in str(x)) and ("science" in str(x))) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "Teacher") if ("teach" in str(x)) or
                                                                                               ("Edu." in str(x)) or (
                                                                                                           "education" in str(
                                                                                                       x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "management information systems")
    if ("management information" in str(x)) or ("bilisim" in str(x)) or (("info" in str(x)) and ("sys" in str(x)))
       or (("bilgi" in str(x)) and ("sis" in str(x))) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "Language")
    if ("Language" in str(x)) or ("language" in str(x)) or ("Dil" in str(x)) or ("langu" in str(x)) or (
                "ingil" in str(x)) or ("dil" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "Math")
    if ("math" in str(x)) or ("matem" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "economy")
    if ("econom" in str(x)) or ("ekono" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "history")
    if ("tarih" in str(x)) or ("hist" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "chemistry")
    if ("kimya" in str(x)) or ("chem" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: x.replace(x, "sayisal")
    if ("mf" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "science") if (("fen" in str(x)) and ("bilim" in str(x))) or (
                    "physical science" in str(x))
                                             or ("natural science" in str(x)) or ("physics" in str(x)) or (
                                                         "fen" in str(x)) or ("fizik" in str(x))
        else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "data") if ("data" in str(x)) or ("artificial intelligence" in str(x)) or
                                          ("machine learning" in str(x)) or ("computer vision" in str(x)) or (
                                                      "nlp" in str(x)) or ("natural language" in str(x)) else x)

    df["fields_of_study2"] = df["fields_of_study2"].apply(
        lambda x: x.replace(x, "data") if (("veri" in str(x)) and ("bilim" in str(x))) else x)

    df["fields_of_study2"] = df["fields_of_study2"].str.lower()

    lst = list(df["fields_of_study2"].value_counts()[0:18].index)

    df["fields_of_study3"] = df["fields_of_study2"].apply(lambda x: str(x).replace(str(x), "other")
    if (str(x) not in lst) else x)

    len(df["fields_of_study"].value_counts())

    # skill

    df["skill2"] = df["skill"].apply(
        lambda x: unidecode.unidecode(str(x)))

    df["skill2"] = df["skill2"].str.lower()

    df["skill3"] = df["skill2"]
    df["skill3"].apply(skill_lambda)

    dfgr = df.groupby(["user_id", "skill3"]).agg({"moved_after_2019": ["mean", "count"]}).sort_values(
        ('moved_after_2019', 'count'), ascending=False)

    dfgr[('moved_after_2019', 'count')] = dfgr[('moved_after_2019', 'count')].apply(lambda x: 1 if x > 1 else x)

    dfgr = dfgr.droplevel(level=0)
    dfgr = dfgr.reset_index()

    dfgr = dfgr.drop(dfgr[dfgr[('moved_after_2019', 'count')] == 0].index, axis=0)

    dfgr = dfgr.groupby("skill3").agg({('moved_after_2019', 'mean'): ["mean", "count"]}).sort_values(
        ('moved_after_2019', 'mean', 'count'), ascending=False)

    dfgr = dfgr[np.logical_and(dfgr[('moved_after_2019', 'mean', 'count')] > 100,
                               (np.logical_or(dfgr[('moved_after_2019', 'mean', 'mean')] > 0.60
                                              , dfgr[('moved_after_2019', 'mean', 'mean')] < 0.30)))]

    lst = list(dfgr[np.logical_or(np.logical_and(dfgr[('moved_after_2019', 'mean', 'mean')] < 0.3,
                                                 dfgr[('moved_after_2019', 'mean', 'count')] > 1000),
                                  dfgr[('moved_after_2019', 'mean',
                                        'mean')] > 0.60)].index)


    for i in lst:
        df[i + " ?"] = df["skill3"].apply(lambda x: 1 if str(i) in str(x) else 0)

    dfs = df.groupby("user_id").apply(lambda x: x["skill3"].nunique()).reset_index()
    del (dfs)
    dfs[dfs["user_id"] == 6950]
    dfs.info()
    df = pd.merge(df, df.groupby("user_id").apply(lambda x: x["skill3"].nunique()).reset_index(), on="user_id",
                  how="left")  # 0 isimli kolon skill_count olarak değişecek
    df["skill_count"] = df[0]


    # (df.groupby("user_id").agg({"skill": "nunique"})[df.groupby("user_id").agg({"skill3": "nunique"})["skill"] == 0])

    # proficiency

    # elementary-limited_working-professional_working-full_professional-native sayısal olarak üstünlük var
    df["proficiency"] = df["proficiency"].apply(proficiency_lambda)

    # language
    # kaç dil varsa o sütun olarak eklendi

    df = pd.merge(df, df.groupby("user_id").apply(lambda x: x["language"].nunique()).reset_index(), on="user_id",
                  how="left")  # 0 isimli kolon language_count olarak değişecek
    df["language_count"] = df[0]


    # şirket nunique

    df = pd.merge(df, df.groupby("user_id").apply(lambda x: x["company_id"].nunique()).reset_index(), on="user_id",
                  how="left")  # 0 isimli kolon company_count olarak değişecek
    df["company_count"] = df[0]


def filling_nan():
    # degree and proficiency

    df["degree"] = df["degree"].fillna(
        df.groupby("language_count")["degree"].transform(lambda x: x.value_counts().idxmax()))

    df["proficiency"] = df["proficiency"].fillna(
        round(df.groupby(["degree", "language"])["proficiency"].transform("mean")))


df.info()
df.dtypes
lst = ["software engineering ?", "oop ?", "object oriented design ?", "software design ?",
       "software project management ?", "uml ?", "telecommunications ?",
       "integration ?", "java enterprise edition ?", "ci/cd ?", "gulp.js ?"]


def convert_int(df):
    for col in df.columns:
        if df[col].nunique() < 10:
            df[col] = df[col].astype("uint8")


def cat_to_num(df):
    lst = ["industry", "school_name", "degree", "fields_of_study", "language", "proficiency", "location"]
    df.drop(lst, axis=1, inplace=True)
    df = pd.get_dummies(df, columns=lst, drop_first=True)
    df.apply(lambda x: x.astype(str).str.lower()).drop_duplicates(keep='first')
    df = df.drop_duplicates()


df["user_id"].nunique()

lst = list(df.columns.str.contains("degree"))

df.keys()
df.info()
df.loc[:, df.columns.str.contains("degree") | df.columns.str.contains("user_id")][
    np.array(df.columns.values)[df.columns.str.contains("degree")]]

"degree" in list(df.columns)
np.array(df.columns.values)[df.columns.str.contains("degree")]
df.loc[:, df.columns.str.contains("degree") | df.columns.str.contains("user_id")]


df.loc[:, df.columns.str.contains("degree")] = df.loc[:, df.columns.str.contains("degree") | df.columns.str.contains(
    "user_id")].groupby("user_id").apply(
    lambda x: 1 if x[col].sum() >= 1 else x[col] for col in
    np.array(df.columns.values)[df.columns.str.contains("degree")])

df.loc[:, df.columns.str.contains("degree")] = df.loc[:, df.columns.str.contains("degree")].apply(
    lambda x: 1 if df.groupbyx.sum() >= 1 else x[col])


df.loc[:, df.columns.str.contains("degree") | df.columns.str.contains("user_id")].apply(
    lambda x: 1 if df.groupby("user_id")[col].sum() >= 1 else 0 for col in
    np.array(df.columns.values)[df.columns.str.contains("degree")])

# df[df["user_id"] == 0]["fields_of_study_sayisal"].sum()

u_id = np.array(df["user_id"].unique())

df[df["user_id"] == 66273]["degree_Lisans Derecesi"].sum()

df["user_id"].value_counts()


arr = np.array(df.columns.values)[
    ~(df.columns.str.contains("degree") | df.columns.str.contains("proficiency") | df.columns.str.contains("language"))]

arr = np.append(arr, "fields_of_study_language")
arr = np.append(arr, "language_count")

len(arr)

arr = np.delete(arr, np.argwhere(arr == "language_count"))
arr = np.delete(arr, np.argwhere(arr == "company_count"))
arr = np.delete(arr, np.argwhere(arr == 'skill_count'))
arr = np.delete(arr, np.argwhere(arr == 'user_id'))
arr = np.delete(arr, np.argwhere(arr == 'moved_after_2019'))

df.to_csv("raw_data5.csv",index_label=False)
print("done")
df[df["user_id"] == 1301]["degree_Lisans Derecesi"].apply(lambda x: 1)


for col in np.array(df.columns.values)[df.columns.str.contains("degree")]:
    ctr = 1
    x = np.empty(0)
    for id in u_id:
        if df[df["user_id"] == id][col].sum() >= 1:
            x = np.append(x, np.array(df[df["user_id"] == id][col].apply(lambda x: 1)))
        else:
            x = np.append(x, np.array(df[df["user_id"] == id][col].apply(lambda x: 0)))
        print(ctr,col)
        ctr = ctr + 1
    df.loc[:, col] = x
    df[col] = df[col].astype("uint8")

lng_pro = np.append(df.columns.values[df.columns.str.contains("proficiency")], (df.columns.values[df.columns.str.contains("language")]))
pro = lng_pro[:4]
lng = lng_pro[4:]

len(lng_pro)
lng_pro = np.delete(lng_pro, np.argwhere(lng_pro == "language_count"))
lng_pro = np.delete(lng_pro, np.argwhere(lng_pro == "fields_of_study_language"))


lng = np.append(lng, "dummy_lng")
pro = np.append(pro, "dummy_pro")

lng_pro = np.append(lng_pro, "dummy_lng")

def return_dummy_variables():
    new_col = df.loc[:,pro].apply(lambda x : 1 if x.sum() == 0 else 0, axis = 1)
    df["dummy_pro"] = new_col
    new_col = df.loc[:, lng].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
    df["dummy_lng"] = new_col


(df[df.index == 0]["dummy_pro"] == 1).values[0]
df.iloc[25000]
df.to_csv("raw_data6.csv",index_label=False)



df.rename(columns = {"dummy_pro":"proficiency_1.0"},inplace=True)

def lang_and_pro_lambda(x):
    cols = df.columns.values[df.columns.str.contains("proficiency")]
    indexes = x[x == 1].index
    return [df.loc[i][cols][df.loc[i][cols] == 1].index.values[0].replace("proficiency_","") if i in indexes else 0 for i in df.index]

df[["language_Turkish"]].apply(lang_and_pro_lambda)

df.loc[:,lng]
def language_and_proficiency_prep():
    #seçilen yeterlilik düzeyi direkt olarak dilin üzerine verilecek örneğin ingilizce sütununa yeterlilik düzeyi basılacak
    #languagede hepsi sıfır ise dummy_lan isminde değişken oluşturulacak ve proficiency
    df[lng] = df[lng].apply(lang_and_pro_lambda)
df[df["user_id"] == 1301]

for col in lng:
    df[col] = df[col].astype("float")

df[df["user_id"] == 66273][lng]
df2 = df.groupby("user_id").apply(lambda x : x[lng].apply(lambda x : x.max() if x.sum() > 5 else x.sum()))
pd.concat([df.drop(lng_pro ,axis = 1).drop_duplicates().sort_values("user_id"), df2], axis = 1)
df.to_csv("last_df.csv", index_label= False)
df = pd.merge(df.drop(lng_pro ,axis = 1).drop_duplicates(), df2, on = "user_id", how = "left")

df2.drop_duplicates()
ctr2 = 1
for col in arr:
    ctr = 1
    x = np.empty(0)
    for id in u_id:
        if df[df["user_id"] == id][col].sum() >= 1:
            x = np.append(x, np.array(df[df["user_id"] == id][col].apply(lambda x : 1)))
        else:
            x = np.append(x, np.array(df[df["user_id"] == id][col].apply(lambda x : 0)))
        print(ctr, ctr2)
        ctr = ctr + 1
    ctr2 = ctr2 +1
    df.loc[:, col] = x
    df[col] = df[col].astype("uint8")

df2 = pd.read_csv("raw_data6.csv")
df2[df2["user_id"] == 66271]

df = pd.read_csv("last_df.csv")
convert_int(df)

df.info()
df.dtypes
def grab_num_cols(df):
    lst = []
    for col in df.columns:
        if df[col].nunique() > 6 or "language_count" in col:
            lst.append(col)
    lst.remove("user_id")
    return lst

num_cols = grab_num_cols(df)
df["language_count"].value_counts()


def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outliers(df, cols, plot = False):
    outliers = []
    for col in cols:
        low_limit, up_limit = outlier_thresholds(df,col)
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            outliers.append(col)
            if plot:
                print(df[(df[col] > up_limit) | (df[col] < low_limit)][col])
    return outliers

def replace_with_thresholds(df, o_cols):
    for col in o_cols:
        low_limit, up_limit = outlier_thresholds(df,col)
        df.loc[(df[col] < low_limit), col] = low_limit
        df.loc[(df[col] > up_limit), col] = up_limit

outlier_cols = check_outliers(df,num_cols,True)
replace_with_thresholds(df, outlier_cols)

def visualizing_num_cols(df, cols):
    for col in cols:
        plt.xlabel(col)
        plt.hist(df[col])
        plt.show()
visualizing_num_cols(df, grab_num_cols(df))

def scaling(df):
    scaler = MinMaxScaler()
    dff = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return dff
x = scaling(df)
# [1 if df[df[u_id] == id][col].sum() >= 1 else 0 for id in u_id for col in np.array(df.columns.values)[df.columns.str.contains("degree")]]



df.to_csv("scaled_df.csv")

df["fields_of_study2"] = df["fields_of_study2"].apply(lambda x: np.nan if "nan" in str(x) else x).value_counts()


# df[df["fields_of_study2"].str.contains("machine learning")]["fields_of_study2"].value_counts()




df[df["fields_of_study2"].notna()][
    np.logical_or(df["fields_of_study2"].str.contains("programci"), df["fields_of_study2"].str.contains("programla"))][
    "fields_of_study2"].value_counts()

df[np.logical_and(df["fields_of_study2"].notna(), (
    np.logical_and(df["fields_of_study2"].str.contains("info"), df["fields_of_study2"].str.contains("sys"))))][
    "fields_of_study2"].value_counts()

# skiller kaç taneyse eklenecekdf["skill"].value_counts()
df.groupby("skill").agg({"skill": "count"})

df[df["skill2"].astype(str).str.contains("[*]")]["skill2"].value_counts()

df["skill3"].value_counts()

len(df["fields_of_study2"].value_counts())
df["fields_of_study3"].value_counts().sum() / len(df)
len(df["fields_of_study3"].value_counts())

len(df[df["start_year_month"].gt(201812)].index)
df[df["start_year_month"].gt(201812)]["moved_after_2019"].mean()
len(df["start_year_month"].gt(201812).index)
df = df.drop(df[df["start_year_month"].gt(201812)].index, axis=0)


# df["degree2"] = df["degree2"].apply(lambda x : np.NaN if "nan" in x else x)

df.drop("degree3", axis=1, inplace=True)


"""
location:İstanbul geçenlerin hepsi Istanbul, Turkey olarak değişecek(Ankara,İzmir,Kocaeli) iki tane virgül varsa sadece ikinci kelimeyi seç-->done
greater varsa greateri sil --> done
içinde Turkey geçen sıklığı düşükleri(İzmir, İstanbul, Ankara, (Turkey)) Turkey içine ata diğer kalanlarını yurt dışı yapılacak -->done
bazı şehirler başında boşluk alarak yazılmış bunların başındaki boşluğu sil sonuna Turkey ekle -->done


en başta düşük frekansları ele ve Other yap yüzde 0.5'ten düşük olanları --> done
okullar için kelimeyi ingilizce kodla (unidecode) ve Universitesi -- University olarak değiştirelecek okul iş değiştirme için rol oynuyor olabilir --> done

language için ingilizce, türkçe, almanca, rusça, japonca, fransızca, ispanyolca, arapça, italyanca, çince, latince, azeri türkçesi ve hollanda kategorileri oluşturulacak

industry için yüzde 0.78den düşük değerler other olarak etiketlendi

fields_of_study 

"""

# df.groupby("degree2").agg({"moved_after_2019":["mean","count"]}).sort_values(('moved_after_2019', 'count'), ascending = False)
# df["degree3"] = df["degree2"].apply(lambda x : x.replace(x,"Other") if ("Lisans" not in x) and ("Doktora" not in x) and ("Lise" not in x) and ("Ön lisans" not in x) else x)
# df.groupby("degree3").agg({"moved_after_2019":["mean","count"]}).sort_values(('moved_after_2019', 'count'), ascending = False)

df["fields_of_study2"].value_counts()[0:100]

df["industry"].value_counts()[100 * df["industry"].value_counts() / len(df) > 0.1]  # 0.5 threshold

df["industry"] = df["industry"].apply(
df["industry"].value_counts()[100 * df["industry"].value_counts() / len(df) > 0.78].sum() / len(df))  # 0.78 threshold


# bu değerlerin toplamı yüzde seksenin üzerinde olana kadar seç
