import pandas as pd
import seaborn as sns


df=sns.load_dataset("titanic")

df["sex"].value_counts()
df.info()


dict={col:df[col].nunique() for col in df.columns}

#df["sex"].dtype in ["float","int","object"]


df[["pclass","parch"]].nunique()
df["embarked"]=df["embarked"].astype("category")
df["embarked"].dtype


df[df["embarked"]=="C"]

df[df["embarked"]!="S"]


df[(df["age"]<30) & (df["sex"]=="female")]


df.loc[(df["fare"]>500) | (df["age"]>70)]


df["age"].isnull().sum()
dict2={col : df[col].isnull().sum() for col in df.columns}

df.drop("who",axis=1,inplace=True)


type(df["deck"].mode()[0])
df["deck"].fillna(df["deck"].mode()[0],inplace=True)
df["deck"].isnull().sum()

df["age"].fillna(df["age"].median(),inplace=True)
df["age"].isnull().sum()

df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})



def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"]=df["age"].apply(age_30)

dff=sns.load_dataset("Tips")


dff.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

dff.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]})

dff[(dff["time"] =="Lunch") & (dff["sex"]=="Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                          "tip": ["sum","min","max","mean"]})

dff.loc[(dff["size"] < 3) & (dff["total_bill"] > 10)]["total_bill"].mean()

#dff.loc[(dff["size"] < 3) & (dff["total_bill"] > 10),"total_bill"].mean() ikinci yol

dff["total_bill_tip_sum"]=dff["total_bill"] + dff["tip"]

dff.loc[:,"totalbilltop"] = dff["total_bill"] + dff["tip"]#ikinci yol


f_avg = dff[dff["sex"]=="Female"]["total_bill"].mean()
m_avg = dff[dff["sex"]=="Male"]["total_bill"].mean()

def sex_mean(sex,total_bill):

    if sex == "Female":
        if f_avg < total_bill:
            return 0
        else:
            return 1
    else:
        if m_avg < total_bill:
            return 0
        else:
            return 1

def func(x):

    if x["sex"] == "Female":
        if f_avg < x["total_bill"]:
            return 0
        else:
            return 1
    else:
        if m_avg < x["total_bill"]:
            return 0
        else:
            return 1


dff["total_bill_flag"]= dff.apply(lambda x: sex_mean(x["sex"],x["total_bill"]),axis=1)


dff["total_bill_flag2"]=dff[["sex","total_bill"]].apply(lambda x:sex_mean(x["sex"],x["total_bill"]),axis=1)


dff["total_bill_flag"]= dff[["sex","total_bill"]].apply(sex_mean(sex=dff["sex"],total_bill=dff["total_bill"]))



dff["total_bill_flag3"]  = dff[["sex","total_bill"]].apply(func,axis=1)


#total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.



dff.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag":"count"})


dff_new=dff.loc[dff["total_bill_tip_sum"].sort_values(ascending=False).index[0:30]]


dff_new.shape
