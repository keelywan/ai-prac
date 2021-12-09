import data_process
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from joblib import dump

best_rfc_clf = None
best_cross_val_score = 0
best_n = -1
best_criterion = ""
best_min_samples_split = -1
best_class_weight = None

xTr, xTe, yTr, yTe = data_process.get_data_opt_var()

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
        scores = cross_val_score(clf, xTr, yTr, cv = 5, scoring = "balanced_accuracy")
        score = scores.mean()
        if score > best_cross_val_score:
          best_cross_val_score = score
          best_n = n
          best_criterion = criterion
          best_min_samples_split = min_samples_split
          best_class_weight = class_weight
          best_rfc_clf = clf
        print("RFC n=" + str(n) + " / Cross Val Score:", score)
print("------------")
print("Best Cross Val Score:", best_cross_val_score)
print("Best N:", best_n)
print("Best Criterion:", best_criterion)
print("Best Min Samples Split:", best_min_samples_split)

clf=RandomForestClassifier(
          n_estimators=best_n, 
          criterion=best_criterion, 
          min_samples_split=best_min_samples_split,
          class_weight=best_class_weight
        )
clf.fit(xTr, yTr)
preds = clf.predict(xTe)
accuracy = metrics.accuracy_score(yTe, preds)
print("Test Accuracy:", accuracy)

dump(best_rfc_clf, "RFC" + "-" + str(accuracy) + "-" + str(best_n) + "-" + str(best_criterion) + "-" + str(best_min_samples_split) + ".joblib")