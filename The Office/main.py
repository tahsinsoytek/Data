import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_csv("the_office_series.csv",index_col=0)
min_max_scaler = preprocessing.MinMaxScaler()



plt.plot(df.index+1,df["Viewership"])
plt.xlabel("Episode Number")
plt.ylabel("Viewership(Millions)")


plt.show()

df["ScaledRating"] = min_max_scaler.fit_transform(pd.DataFrame(df["Ratings"]))

colors =[]
for value in df["ScaledRating"]:
    if value < 0.25:
        colors.append("red")
    elif value < 0.50:
        colors.append("orange")
    elif value < 0.75:
        colors.append("lightgreen")
    else:
        colors.append("darkgreen")

df["Color"] = colors



df["HasGuests"] = df["GuestStars"].replace(np.nan,0) != 0
sizes = [250 if x else 25 for x in df["HasGuests"]]
df[df["Viewership"] == 22.91]
df.columns
fig = plt.figure(figsize=(25,15))

plt.scatter(df.index+1, df["Viewership"],s = np.array(sizes),c= colors,alpha=0.75)

plt.xlabel("Episode Number",size = 30)
plt.ylabel("Viewership(Millions)",size = 20)

plt.xticks(size=30)
plt.yticks(size=30)

plt.title("Popularity, Quality, and Guest Appearances on the Office",size=50)

plt.show()



top_stars = df[max(df["Viewership"]) == df["Viewership"]]["GuestStars"]
top_stars = (top_stars.values[0]).split(",")

