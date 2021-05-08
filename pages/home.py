import streamlit as st

import sys
sys.path.append('..')
icon_path = './assets/logo.png'

logo_path = './assets/logo.png'
def render():
    with open('./assets/style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html =True)
    st.image(logo_path)
    st.markdown("<h1 style='text-align: center; color: black;'>Weight Watcher!</h1>", unsafe_allow_html=True)
    learn_more = st.button('Learn More!')
    try_it = st.button('Try it Out!')
    if learn_more:
        return 'Learn More'
    elif try_it:
        return 'Login'
    return 'Home'