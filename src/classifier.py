import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump, load

df_data = pd.read_csv("../mood_data.csv", sep=',',header=None, encoding='unicode_escape', thousands=',')
df_data.columns= df_data.iloc[0]
df_data = df_data.drop(0)

best_clf = None
best_accuracy = 0
best_type = ""
best_variables = []
best_n = -1
best_weight = ""
best_algorithm = ""
best_criterion = ""
best_min_samples_split = -1
best_class_weight = None

# selecting which variables to include - removed time_signature, mode, key
variables = ['acousticness','danceability','energy','instrumentalness','loudness','speechiness','tempo','valence']
X = df_data[variables]
y = df_data['mood']

# split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# max_features{“auto”, “sqrt”, “log2”}

for test_size in [0.2, 0.25, 0.3, 0.35, 0.4]:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  # Random Forest Classifier
  for class_weight in [None, "balanced", "balanced_subsample"]:
    print("RFC " + str(class_weight) + " ================================")
    for split in range(1, 5):
      print("RFC " + str(split) + " =================")
      min_samples_split = split * 2
      for criterion in ["gini", "entropy"]:
        print("RFC " + criterion + " =================")
        for i in range(5, 50):
          n = i * 10
          clf=RandomForestClassifier(
            n_estimators=n, 
            criterion=criterion, 
            min_samples_split=min_samples_split,
            class_weight=class_weight
          )
          clf.fit(X_train,y_train)
          y_pred=clf.predict(X_test)
          accuracy = metrics.accuracy_score(y_test, y_pred)
          if accuracy > best_accuracy:
            best_type = "Random Forest Classifier"
            best_accuracy = accuracy
            best_n = n
            best_variables = variables
            best_weight = ""
            best_algorithm = ""
            best_criterion = criterion
            best_min_samples_split = min_samples_split
            best_class_weight = class_weight
            best_clf = clf
          print("RFC n=" + str(n) + " / Accuracy:", accuracy)
  print("------------")

  # KNN Classifier
  for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
    print("KNN " + algorithm + " ================================")
    for weight in ["uniform", "distance"]:
      print("KNN " + weight + " =================")
      for i in range(1, 20):
        n = i * 10
        clf = KNeighborsClassifier(n_neighbors=n, algorithm=algorithm, weights=weight)
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
          best_type = "KNN"
          best_accuracy = accuracy
          best_n = n
          best_variables = variables
          best_weight = weight
          best_algorithm = algorithm
          best_criterion = None
          best_min_samples_split = None
          best_class_weight = None
          best_clf = clf
        print("KNN n=" + str(n) + " / Accuracy:", accuracy)
print("------------")
print("Best Type:", best_type)
print("Best Variables:", best_variables)
print("Best N:", best_n)
print("Best Accuracy:", best_accuracy)
print("Best Criterion:", best_criterion)
print("Best Min Samples Split:", best_min_samples_split)
dump(best_clf, str(best_accuracy) + "-" + best_type + "-" + str(best_n) + "-" + best_criterion + "-" + best_min_samples_split + ".joblib")