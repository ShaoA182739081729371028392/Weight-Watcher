import streamlit as st
import sys
sys.path.append('..')
path_logo = './assets/logo.png'
def render():
    st.title("Weight Watcher: ")
    st.header("Start living a healthier life today!")
    st.image(path_logo, width = 200)
    st.markdown("Living a healthier life is a challenging task, and it all starts from one step: Eating better. However, keeping track of one\'s nutrition is a very onerous task, making many give up prematurely. Weight Watcher solves this problem, leveraging Deep Learning and Computer Vision to estimate the weight and calories of your meals, allowing you to count your calories simply from an image of food! Not only that, Weight Watcher also monitors your progress, allowing you to see your improvements in your lifestyle. Register today, and start living a healthier life in two clicks!")