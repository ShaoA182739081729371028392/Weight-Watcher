import streamlit as st
import sys
sys.path.append('..')
icon_path = './assets/logo.png'

source_code_link = 'https://github.com/ShaoA182739081729371028392/Weight-Watcher'

MENU = [
    'Home', # Simply a Home page, with project details
    'Login',
    'Learn More' # Sophisticated part.
]


def render():
    # Initialize Main Bar
    cur_page = 'Learn More'
    st.sidebar.title("Weight Watcher Menu")
    st.sidebar.image(icon_path)
    for header in MENU:
        chosen = st.sidebar.button(header)
        if chosen:
            cur_page = header
    st.sidebar.info(f"All Source Code can be found [here]({source_code_link})")
    st.title("Weight Watcher: ")
    st.header("Start living a healthier life today!")
    st.image(icon_path, width = 200)
    st.markdown("Living a healthier life is a challenging task, and it all starts from one step: Eating better. However, keeping track of one\'s nutrition is a very onerous task, making many give up prematurely. Weight Watcher solves this problem, leveraging Deep Learning and Computer Vision to estimate the weight and calories of your meals, allowing you to count your calories simply from an image of food! Not only that, Weight Watcher also monitors your progress, allowing you to see your improvements in your lifestyle. Register today, and start living a healthier life in two clicks!")
    #st.video()
    return cur_page