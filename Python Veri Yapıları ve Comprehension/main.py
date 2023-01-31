x=8
y=3.2
z=8j+18
a="hello world"
b=True
c=23<22
l=[1,2,3,4]
d={"Name":"Jake"}
t=("Machine Learning","Data Science")
s={"python","machine learning"}

type(x)
type(y)
type(z)
type(a)
type(b)
type(c)
type(l)
type(d)
type(t)
type(s)

text="The goal is to turn data into information, and information into insight"
text=text.upper()
text=(text.replace(","," ")).split()

lst=["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)
lst[0]
lst[10]
lst2=lst[0:4]
lst.pop(8)
lst.append("X")
lst.insert(8,"N")


dict={"Christian":["America",22],
      "Daisy":["England",12],
      "Antonio":["Spain",22],
      "Dante":["Italy",25]}

dict.keys()
dict.values()
#dict["Daisy"]=["England",13]  Alternatif çözüm
dict.update({"Daisy":["England",13]})
dict.update({"Ahmet":["Turkey",24]})
dict.pop("Antonio")


l=[2,13,18,93,22]

def func(lst):

    e_list,o_list=[],[]
    for x in l:
        if x%2==0:
            e_list.append(x)
        else:
            o_list.append(x)

    return e_list,o_list
even,odd=func(l)

import seaborn as sns

df=sns.load_dataset("car_crashes")
df
df.info()


df.columns=["NUM_"+col.upper() if df[col].dtype!="O" else col.upper() for col in df.columns]

df.columns=[col.upper()+"_FLAG" if "no" not in col else col.upper() for col in df.columns]

og_list=["abbrev", "no_previous"]

df_new=[col for col in df.columns if col not in og_list]

import numpy as np

list=np.random.randint(0,100,size=30)
np.random.randint(0,5)

#sayıların karekökünü key olarak yaz tabi sayı çiftse
dict = {x ** 0.5 : x for x in list if x%2==0}

#rastgele dict oluştur ve key bunun kaçıncı kuvvetini alacağını göstersin sayı tekse üstten çiftse alttan alsın
dict2 = {np.random.randint(0,765):np.random.randint(1,4)  for i in range(5)}


list_comp = [k**(v) if k % 2 != 0 else k**(1/v) for (k,v) in dict2.items()]