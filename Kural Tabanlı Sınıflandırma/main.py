
"""
PRICE – Müşterinin harcama tutarı
Değişkenler
SOURCE – Müşterinin bağlandığı cihaz türü
SEX – Müşterinin cinsiyeti
COUNTRY – Müşterinin ülkesi
AGE – Müşterinin yaşı
persona.csv
İ Ş PROBLEMİ
VERİ SETİ
DEĞİŞKENLER
GÖREVLER
"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_csv("persona.csv")


df.shape
df.head()
df.info()


df["SOURCE"].nunique()
df["SOURCE"].value_counts()


df["PRICE"].nunique()
df["PRICE"].value_counts()



df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()
df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")



df.groupby("COUNTRY").agg({"PRICE":"sum"})

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")



#soru 7
df.groupby("SOURCE").agg({"PRICE":"count"})
df["SOURCE"].value_counts()



#soru 8


df.groupby("COUNTRY").agg({"PRICE":"mean"})


#soru 9

df.groupby("SOURCE").agg({"PRICE":"mean"})

#soru 10

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})


#görev 2

df2=df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})

df.pivot_table(values="PRICE",index=["COUNTRY", "SOURCE", "SEX", "AGE"],aggfunc="mean").head()

#görev 3

agg_df=df2.sort_values("PRICE",ascending=False)

x=agg_df.reset_index()#indexleri alıp sütuna atıyoruz
x.shape
import numpy as np
x.index=x["PRICE"]
x.index=x["COUNTRY"]
x.index = np.arange(0, len(x), 1)
x["PRICE"]=x.index

x.drop("COUNTRY",axis=1,inplace=True)
x.reset_index(inplace=True)

x.index


#görev 5
agg_df.reset_index(inplace=True)


bins = [0,18,23,30,40,agg_df["AGE"].max()]

mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]


agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins,labels=mylabels)




# görev 6

agg_df["customers_level_based"] = [row[0].upper()+"_"+row[1].upper()+"_"+row[2].upper()+"_"+row[3] for row in agg_df.drop(["AGE", "PRICE"], axis=1).values]






agg_df = agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})
agg_df.reset_index(inplace=True)
agg_df["customers_level_based"].value_counts()


# YÖNTEM 1

agg_df["customers_level_based2"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]

# YÖNTEM 2
agg_df['customers_level_based3'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x).upper(), axis=1)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D","C","B","A"])

agg_df.groupby("SEGMENT").agg({"PRICE":["mean", "max", "sum"]})


#33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

#35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_31_40"


agg_df[agg_df["customers_level_based"] == new_user]

agg_df.loc[agg_df["customers_level_based"] == new_user]











