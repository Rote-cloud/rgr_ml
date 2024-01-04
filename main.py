import streamlit as st
from web_page1 import *
from web_page2 import *
from web_page3 import *
from web_page4 import *
from web_page5 import *

PAGES = [
    "Информация о разработчике",
    "Информация о наборе данных",
    "Визуализация зависимостей",
    "Получить качество обученных моделей",
    "Сделать предсказание на основе введённых данных"
]

page = st.sidebar.radio("Навигация", PAGES, index=0)

if page == "Информация о разработчике":
    web_page1()
elif page == "Информация о наборе данных":
    web_page2()
elif page == "Визуализация зависимостей":
    web_page3()
elif page == "Получить качество обученных моделей":
    web_page4()
elif page == "Сделать предсказание на основе введённых данных":
    web_page5()

