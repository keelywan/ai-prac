import data_process
from sklearn.model_selection import cross_val_score
from sklearn import svm
from joblib import dump

best_svm_clf = None
best_cross_val_score = 0
best_c = -1
best_gamma = None
best_kernel = None
best_class_weight = None

X, Y = data_process.get_data_opt_var()

# SVM Classifier
for gamma in ["scale", "auto"]:
    print("SVM " + str(gamma) + " ================================")
    for c in range(-2, 10):
        print("SVM " + str(c) + " ================================")
        for kernel in ["linear", "rbf"]:
            print("SVM " + str(kernel) + " ================================")
            for class_weight in [None, "balanced"]:
                print("SVM " + str(class_weight) + " ================================")
                clf = svm.SVC(C=2**c, gamma=gamma, class_weight=class_weight,kernel=kernel)
                scores = cross_val_score(clf, X, Y, cv = 5, scoring = "balanced_accuracy")
                score = scores.mean()
                if score > best_cross_val_score:
                    best_cross_val_score = score
                    best_kernel = kernel
                    best_c = c
                    best_gamma = gamma
                    best_class_weight = class_weight
                    best_svm_clf = clf
                print("SVM details " + str(c) + str(gamma) + kernel + str(class_weight) + " / Accuracy:", score)
    
print("Best Accuracy:", best_cross_val_score)            
print("Best C:", str(best_c))
print("Best Gamma:", best_gamma)
print("Best Kernel:", best_kernel)
print("Best Class Weight:", best_class_weight)
dump(best_svm_clf, "SVM" + "-" + str(best_cross_val_score) + "-" + str(best_c) + "-" + str(best_gamma) + "-" + str(best_kernel) + "-" + str(best_class_weight) + ".joblib")