import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

def web_page5():
    st.title("На основе данных сделать предсказание задержки рейса")

    df = pd.read_csv('./df/airlines_task.csv')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # st.header("Авиакомпания")
    airline_unique_value = list(df['Airline'].unique())
    airlines = st.selectbox("Авиакомпания", airline_unique_value)

    # st.header("Номер полёта")
    flight = st.number_input("Номер полёта", min_value=0, max_value=10000, value=0)

    # st.header("Аэропорт отбытия")
    airportfrom_unique_value = list(df['AirportFrom'].unique())
    airportfrom = st.selectbox("Отбытие", airportfrom_unique_value)

    # st.header("Аэропорт прибытия")
    airportto_unique_value = list(df['AirportTo'].unique())
    airportto = st.selectbox("Прибытие", airportto_unique_value)

    # st.header("День недели вылета")
    dayOfWeek = st.number_input("День недели вылета", min_value=1, max_value=7, value=1)

    time = st.number_input("Длительность полёта", min_value=0, max_value=10000, value=0)

    length = st.number_input("Длина полёта", min_value=0, max_value=1000, value=0)

    data = {'Flight': [flight], 'AirportFrom': [airportfrom_unique_value.index(airportfrom)],
            'AirportTo': [airportto_unique_value.index(airportto)], 'FayOfWeek': [dayOfWeek],
            'Time': [time], 'Length': [length]}
    df = pd.DataFrame(data)

    for airline in airline_unique_value:
        if airline == airlines:
            df[airline] = 1
        else:
            df[airline] = 0

    X = df.values.flatten()
    X = X.reshape(1, -1)

    button = st.button("Сделать предсказание")

    if button:
        st.header("1 - рейс будет задержан")
        st.header("0 - рейс не задержат")

        with open('./models/knn.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write(f'knn: {model.predict(X)[0]}')

        with open('./models/bagging.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write(f'bagging: {model.predict(X)[0]}')

        with open('./models/boosting.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write(f'boosting: {model.predict(X)[0]}')

        with open('./models/stacking.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write(f'stacking: {model.predict(X)[0]}')

        model = tf.keras.models.load_model('models/mlp.h5')
        st.write(f'mlp: {round(model.predict(X)[0][0])}')

