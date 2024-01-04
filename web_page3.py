import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def web_page3():
    file = st.file_uploader("Выберите файл датасета", type=["csv", "xlsx", "txt"])

    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, sep='\t')

        st.write("Полученный набор данных:", df)

        st.write("Визуализация зависимостей:")

        st.header("Гистограммы")
        column = ["Flight", "Time", "Length"]

        for col in column:
            plt.figure(figsize=(10, 8))
            sns.histplot(df[col], bins=100)
            plt.title(f'{col}')
            st.pyplot(plt)

        st.header("Карта корреляции:")
        plt.figure(figsize=(20, 16))
        sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title('Карта с корреляцией')
        st.pyplot(plt)

        st.header("Диаграммы размаха")

        for col in column:
            plt.figure(figsize=(10, 8))
            sns.boxplot(df[col])
            plt.title(f'{col}')
            st.pyplot(plt)

        st.header("Круговые диаграммы")
        column = ["DayOfWeek", "Airline"]

        for col in column:
            plt.figure(figsize=(10, 10))
            df[col].value_counts().plot.pie(autopct='%1.1f%%')
            plt.title(f'{col}')
            st.pyplot(plt)