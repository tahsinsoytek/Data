import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import  mean_absolute_percentage_error,mean_absolute_error
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.precision', 4)


def analysis_scaling_methods():
    #verilere farklı ölçeklendirme yöntemleri uygulayarak performans farklarına bakılacak
    df = pd.read_csv("last_df_encoded.csv")
    df = df[df["dağıtılan enerji (mwh)"] != 0]
    X = df.drop(["dağıtılan enerji (mwh)","tarih"],axis = 1)
    y = df["dağıtılan enerji (mwh)"]
    num_col = ["temp"]
    X_standart_scaler = StandardScaler().fit_transform(df[num_col])#Standart Scaler daha iyi sonuç verdi.
    X_min_max = MinMaxScaler().fit_transform(df[num_col])
    X_robust = RobustScaler().fit_transform(df[num_col])

    regressors = [('LR', LinearRegression()),  # neg_mean_absolute_error: -217.4686 (LR)
                  ('KNN', KNeighborsRegressor()),  # neg_mean_absolute_error: -201.1382 (KNN)
                  ("CART", DecisionTreeRegressor()),  # neg_mean_absolute_error: -200.9604 (CART)
                  ("RF", RandomForestRegressor()),  # neg_mean_absolute_error: -186.5649 (RF)
                  ('Adaboost', AdaBoostRegressor()),
                  # neg_mean_absolute_error: -228.6752 (Adaboost) -----> ÖLÇEKLENDİRMEDEN ÖNCEKİ DEĞERLER STANDARD SCALER DEĞERLERİ DÜŞÜRÜYOR.
                  ('GBM', GradientBoostingRegressor()),  # neg_mean_absolute_error: -179.9774 (GBM)
                  ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
                  # neg_mean_absolute_error: -181.5666 (XGBoost)
                  ('LightGBM', LGBMRegressor()),  # neg_mean_absolute_error: -183.0553 (LightGBM)
                  ]

    for name, regressor in regressors:

        scoring = "neg_mean_absolute_error"

        print("Standard Scaler")
        X["temp"] = X_standart_scaler
        cv_results = cross_validate(regressor, X, y, cv=10, scoring = scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

        print("Min Max")
        X["temp"] = X_min_max
        cv_results = cross_validate(regressor, X_min_max, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

        print("Robust")
        X["temp"] = X_robust
        cv_results = cross_validate(regressor, X_robust, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")


knn_params = {"n_neighbors": range(10, 30)}

cart_params = {'max_depth': range(10, 20),
               "min_samples_split": range(20, 35)}

rf_params = {"max_depth": [8, 12, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20, 35, 50],
             "n_estimators": [200, 300, 500]}


gbm_params = {"max_depth": [5, 8, 12, 20],
              "learning_rate": [0.01, 0.05, 0.1],#daha yüksek çıktı
              "n_estimators": [100, 200, 500, 1000],
              "subsample": [0.6, 0.8, 1]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 10],
                  "n_estimators": [100, 200, 450, 600],
                  "colsample_bytree": [0.5, 0.7, 1]}



lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 700],
                   "colsample_bytree": [0.5, 0.7, 1]}


regressors = [
              ('KNN', KNeighborsRegressor(), knn_params),  # neg_mean_absolute_error: -201.1382 (KNN)
              ("CART", DecisionTreeRegressor(), cart_params),  # neg_mean_absolute_error: -200.9604 (CART)
              ("RF", RandomForestRegressor(), rf_params),  # neg_mean_absolute_error: -186.5649 (RF)
              ('GBM', GradientBoostingRegressor(), gbm_params),  # neg_mean_absolute_error: -179.9774 (GBM)
              ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgboost_params),# neg_mean_absolute_error: -181.5666 (XGBoost)
              ('LightGBM', LGBMRegressor(), lightgbm_params),  # neg_mean_absolute_error: -183.0553 (LightGBM)
              ]

def hyperparameter_optimization(X, y, cv=5, scoring="neg_mean_absolute_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


random_knn_params = {"n_neighbors": range(2, 40)}

random_cart_params = {'max_depth': range(1, 30),
                      "min_samples_split": range(2, 30)}

random_rf_params = {"max_depth": [30, 35, 45, 70, 100, None],
                    "max_features": [10, 12, 15, 25, 40, 60, "auto"],#rf değerleri sınırda model daha kompleks hale gelecek
                    "min_samples_split": [10, 20, 25, 30, 40, 60, 80],
                    "n_estimators": [200, 300, 400, 700, 1000, 2000, 3000,5000]}


random_gbm_params = {"max_depth": [5, 8, 10, 20, 30, 40, 70, 100],
                     "learning_rate": [0.01, 0.05, 0.1, 0.2],
                     "n_estimators": [200, 300, 400, 700, 1000, 2000, 3000,5000],
                     "subsample": [0.6, 0.8, 1]}

random_xgboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.2],
                         "max_depth": [5, 8, 10, 20, 30, 40, 70, 100],
                         "n_estimators": [200, 300, 400, 700, 1000, 2000, 3000,5000],
                         "colsample_bytree": [0.5, 1]}

random_lightgbm_params = {"max_depth": [5, 8, 10, 20, 30, 40, 70, 100],
                          "learning_rate": [0.01, 0.05, 0.1, 0.2],
                          "n_estimators": [200, 300, 400, 700, 1000, 2000, 3000,5000],
                          "colsample_bytree": [0.3, 0.5, 0.7, 1]}

regressors = [
              ('KNN', KNeighborsRegressor(), random_knn_params),  # neg_mean_absolute_error: -201.1382 (KNN)
              ("CART", DecisionTreeRegressor(), random_cart_params),  # neg_mean_absolute_error: -200.9604 (CART)
              ("RF", RandomForestRegressor(), random_rf_params),  # neg_mean_absolute_error: -186.5649 (RF)
              ('GBM', GradientBoostingRegressor(), random_gbm_params),  # neg_mean_absolute_error: -179.9774 (GBM)
              ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), random_xgboost_params),# neg_mean_absolute_error: -181.5666 (XGBoost)
              ('LightGBM', LGBMRegressor(), random_lightgbm_params),  # neg_mean_absolute_error: -183.0553 (LightGBM)
              ]

def randomized_hyperparameter_optimization(X, y, cv=5, scoring="neg_mean_absolute_error"):
    print("Randomized Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        rs_best = RandomizedSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**rs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_reg = VotingRegressor(estimators = [('KNN', best_models["KNN"]), ('CART', best_models['CART']), ('RF', best_models["RF"]), ('GBM', best_models['GBM']), ('XGBoost', best_models['XGBoost']),
                                              ('LightGBM', best_models["LightGBM"])]).fit(X, y)
    cv_results = cross_validate(voting_reg, X, y, cv=3, scoring = "neg_mean_absolute_error")
    print(f"MAE: {cv_results['test_score'].mean()}")
    return voting_reg

def main():
    df = pd.read_csv("last_df_encoded.csv")
    num_col = ["temp"]
    X_standart_scaler = StandardScaler().fit_transform(df[num_col])
    df["temp"] = X_standart_scaler
    X = df[df["dağıtılan enerji (mwh)"] != 0]
    y = X["dağıtılan enerji (mwh)"]
    X = X.drop(["dağıtılan enerji (mwh)", "tarih"], axis=1)

    Z = X.drop(["yıl_2019.0","yıl_2020.0","yıl_2021.0","yıl_2022.0"], axis = 1)

    best_models = hyperparameter_optimization(Z, y)
    voting_reg = voting_regressor(best_models, Z, y)


    joblib.dump(voting_reg, "voting_reg.pkl")
    return voting_reg
