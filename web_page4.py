import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, completeness_score
from sklearn.metrics import classification_report
import tensorflow as tf
import pickle
import numpy as np

def predict(clf, X_test, y_test):
    y_pred = np.round(clf.predict(X_test))

    st.write(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    st.write(f'Precision: {precision_score(y_test, y_pred)}')
    st.write(f'Recall: {recall_score(y_test, y_pred)}')
    st.write(f'F1: {f1_score(y_test, y_pred)}')
    st.write(f'Roc: {roc_auc_score(y_test, y_pred)}')

def predictClaster(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    st.write(f"Silhouette Score: {silhouette_score(y_test, y_pred)}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_score(y_test, y_pred)}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_score(y_test, y_pred)}")
    st.write(f"Adjusted Rand Index: {adjusted_rand_score(y_test, y_pred)}")
    st.write(f"Completeness Score: {completeness_score(y_test, y_pred)}")

def web_page4():
    file = st.file_uploader("Выберите файл датасета", type=["csv", "xlsx", "txt"])

    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, sep='\t')

        st.write("Полученный набор данных:", df)

        models = ["knn", "km", "bagging", "stacking", "boosting", "mlp"]
        select_model = st.selectbox("Выберите тип модели", models)

        button = st.button("Обучить модель")

        if button:
            df.drop(['id'], axis=1, inplace=True)

            enc = OrdinalEncoder()
            df[['AirportFrom', 'AirportTo']] = enc.fit_transform(df[['AirportFrom', 'AirportTo']])

            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            enc = OneHotEncoder(handle_unknown='ignore')
            df_enc = pd.DataFrame(enc.fit_transform(df[['Airline']]).toarray(), columns=df['Airline'].unique())

            new_df = df.join(df_enc)
            new_df.drop(['Airline'], axis=1, inplace=True)

            X = new_df.drop(['Delay'], axis=1)
            scaler = StandardScaler()
            scaler_x = scaler.fit_transform(X)
            y = new_df['Delay']

            nm = NearMiss()
            new_X, new_y = nm.fit_resample(scaler_x, y.ravel())

            X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)


            if select_model == "knn":
                with open('./models/knn.pkl', 'rb') as file:
                    model = pickle.load(file)
                predict(model, X_test, y_test)
            elif select_model == "km":
                with open('./models/km.pkl', 'rb') as file:
                    model = pickle.load(file)
                predictClaster(model, X_test, y_test)
            elif select_model == "bagging":
                with open('./models/bagging.pkl', 'rb') as file:
                    model = pickle.load(file)
                predict(model, X_test, y_test)
            elif select_model == "boosting":
                with open('./models/boosting.pkl', 'rb') as file:
                    model = pickle.load(file)
                predict(model, X_test, y_test)
            elif select_model == "stacking":
                with open('./models/stacking.pkl', 'rb') as file:
                    model = pickle.load(file)
                predict(model, X_test, y_test)
            elif select_model == "mlp":
                model = tf.keras.models.load_model('models/mlp.h5')
                predict(model, X_test, y_test)