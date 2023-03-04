import pandas as pd
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import re
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")                                       #roc_auc: 0.7387 (LightGBM)
    classifiers = [('LR', LogisticRegression()),                 #roc_auc: 0.6348 (LR)
                   ('KNN', KNeighborsClassifier()),                 #roc_auc: 0.6534 (KNN)
                   ("SVC", SVC()),                                  #roc_auc: 0.6895 (SVC)
                   ("CART", DecisionTreeClassifier()),               #roc_auc: 0.6531 (CART)
                   ("RF", RandomForestClassifier()),                    #roc_auc: 0.8140 (RF)
                   ('Adaboost', AdaBoostClassifier()),                       #roc_auc: 0.6874 (Adaboost)
                   ('GBM', GradientBoostingClassifier()),                       #roc_auc: 0.7034 (GBM)
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')), #roc_auc: 0.7451 (XGBoost)
                   ('LightGBM', LGBMClassifier()),
                   ]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")



cart_params_r = {'max_depth': range(1,30),
               "min_samples_split": range(2,30)}

rf_params_r = {"max_depth": [8, 15, 30, 50, None],  #default değerler çevresinde dolaşılacak
             "max_features": [5, 7, 10, 15, 20 , "auto"],
             "min_samples_split": [15, 20, 30, 40, 50],
             "n_estimators": [200, 300, 500, 750, 1000, 2000,3500, 5000]}

xgboost_params_r = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 10, 20, 30,50],   #'n_estimators': 300, 'max_depth': 50, 'learning_rate': 0.01, 'colsample_bytree': 0.5
                  "n_estimators": [100, 200, 300, 500, 750, 1000, 2000,3500, 5000],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params_r = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [100, 200, 300, 500, 750, 1000, 2000,3500, 5000],#'n_estimators': 5000, 'learning_rate': 0.01, 'colsample_bytree': 0.7
                   "colsample_bytree": [0.7, 1]}

classifiers = [ ("CART", DecisionTreeClassifier(), cart_params_r),
               ("RF", RandomForestClassifier(), rf_params_r),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params_r),
               ('LightGBM', LGBMClassifier(), lightgbm_params_r)]



def random_search(X, y, cv = 3, scoring="roc_auc"):
    dct = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        rs_best = RandomizedSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**rs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")
        dct[name] = rs_best.best_params_
    return dct



rf_params = {"max_depth": [8, 15, 40, None],
             "max_features": [50, 75, 35,"auto"],
             "min_samples_split": [2, 7, 10],
             "n_estimators": [150,100,250,400]}

xgboost_params = {"learning_rate": [0.05 ,0.03 ,0.01, 0.005],
                  "max_depth": [40, 50, 60],
                  "n_estimators": [250,300,350,400],#'n_estimators': 300, 'max_depth': 50, 'learning_rate': 0.01, 'colsample_bytree': 0.5
                  "colsample_bytree": [0.5, 0.3, 0.6]}

lightgbm_params = {"learning_rate": [0.05 ,0.03 ,0.01, 0.005],
                   "n_estimators": [4000, 5000, 6000], #'n_estimators': 5000, 'learning_rate': 0.01, 'colsample_bytree': 0.7
                   "colsample_bytree": [0.3, 0.5, 0.7]}


classifiers = [ ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]



def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

"""
########## RF ##########
roc_auc (Before): 0.8218
roc_auc (After): 0.8244
RF best params: {'max_depth': None, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 250}
########## XGBoost ##########
roc_auc (Before): 0.7428
roc_auc (After): 0.8121
XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.03, 'max_depth': 40, 'n_estimators': 250}
########## LightGBM ##########
roc_auc (Before): 0.7383
roc_auc (After): 0.7878
LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'n_estimators': 5000}
"""

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

def best_models(x_train, y_train, x_test, y_test):
    models = {}
    dct = {}
    dct[0] = len(y_train) / (2 * y_train.value_counts()[0])
    dct[1] = len(y_train) / (2 * y_train.value_counts()[1])
    rf_params = {'max_depth': None, 'max_features': int(len(x_train.columns)/4), 'min_samples_split': 2, 'n_estimators': 250, 'class_weight':dct}
    rf = RandomForestClassifier().set_params(**rf_params)
    print(rf.get_params())
    rf.fit(x_train, y_train)
    models["RF"] = rf


    pos_weight = 2 * (y_train.value_counts()[0]/ (y_train.value_counts()[1]))
    xgb_params = {'colsample_bytree': 0.5, 'learning_rate': 0.03, 'max_depth': 40, 'n_estimators': 1000, 'scale_pos_weight': pos_weight }
    xgb = XGBClassifier(use_label_encoder= False).set_params(**xgb_params)
    xgb.fit(x_train, y_train)
    models["XGBoost"] = xgb

    lgbm_params = {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'n_estimators': 5000, 'class_weight':dct}
    lgbm = LGBMClassifier().set_params(**lgbm_params)
    lgbm.fit(x_train, y_train)
    models["LightGBM"] = lgbm


    for model in models.values():
        threshold = selecting_best_th_value(x_train, y_train, model)
        predicted_proba = model.predict_proba(x_test)
        y_pred = (predicted_proba[:, 1] >= threshold).astype('int')
        y_pred = y_pred[0]
        print(f"#######{model}#######")
        print("Threshold value:", threshold[0][0])
        #y_pred = model.predict(x_test)
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred)
        print("roc_auc:", roc_auc_score(y_test, model.predict_proba(x_test)[:,1]))

    return models
def selecting_best_th_value(x_train, y_train, model):
    y_pred_proba = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
    th_values = tpr - fpr
    max_value = th_values.max()
    index = np.argwhere(th_values == max_value)
    return thresholds[index]

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]), ('XGBoost',best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft')
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

def main():
    df = pd.read_csv("scaled_df.csv").dropna().drop("user_id", axis = 1)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    train, test, = train_test_split(df, train_size = 0.7)
    y_train = train.loc[:,"moved_after_2019"]
    x_train = train.drop("moved_after_2019", axis = 1)
    y_test = test.loc[:, "moved_after_2019"]
    x_test = test.drop("moved_after_2019", axis=1)
    base_models(x_train,y_train)
    random_search(x_train,y_train)
    hyperparameter_optimization(x_train, y_train)
    models = best_models(x_train, y_train, x_test, y_test)
    voting_clf = voting_classifier(models, x_test, y_test)
    joblib.dump(voting_clf, "voting_clf.pkl")


print("done")
if __name__ == "__main__":
    print("İşlem başladı")
    main()



