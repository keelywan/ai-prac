import data_process
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

best_knn_clf = None
best_cross_val_score = 0
best_n = -1
best_weight = None
best_algorithm = None

# X, Y = data_process.get_data_opt_var()

# KNN Classifier
for group, X, Y in data_process.powerset_get_data():
    print(group)
    for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
        print("KNN " + algorithm + " ================================")
        for weight in ["uniform", "distance"]:
            print("KNN " + weight + " =================")
            for i in range(1, 20):
                n = i * 10
                clf = KNeighborsClassifier(n_neighbors=n, algorithm=algorithm, weights=weight)
                scores = cross_val_score(clf, X, Y, cv = 5, scoring = "balanced_accuracy")
                score = scores.mean()
                if score > best_cross_val_score:
                    best_cross_val_score = score
                    best_n = n
                    best_algorithm = algorithm
                    best_weight = weight
                    best_knn_clf = clf
                print("KNN n=" + str(n) + " / Accuracy:", score)
print("------------")

print("Best Accuracy:", best_cross_val_score)
print("Best N:", best_n)
print("Best Algorithm:", best_algorithm)
print("Best Weight:", best_weight)
dump(best_knn_clf, "KNN" + "-" + str(best_cross_val_score) + "-" + str(best_n) + "-" + str(best_algorithm) + "-" + str(best_weight) + ".joblib")