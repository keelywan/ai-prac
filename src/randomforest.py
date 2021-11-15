import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df_data = pd.read_csv("../mood_data.csv", sep=',',header=None, encoding='unicode_escape', thousands=',')
df_data.columns= df_data.iloc[0]
df_data = df_data.drop(0)

# selecting which variables to include - removed time_signature, mode, key
X = df_data[['acousticness','danceability','energy','instrumentalness','loudness','speechiness','tempo','valence']]
y = df_data['mood']

# split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf=RandomForestClassifier(n_estimators=150)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))