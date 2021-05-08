'''
Features of the Application:
- BMI
- Goal Tracking
- graphs
- Calorie Tracking
- Food Journal
- Exercise Journal 
- Health Tips
- Login for Privacy
'''

import streamlit as st
import sys
sys.path.append('..')

icon_path = './assets/logo.png'
MENU = [
    'Home', # Simply a Home page, with project details
    'Login',
    'Learn More'
]
def render_not_logged_in():
    st.header("Login or Register with Weight Watcher!")
def render_logged_in(profile):
    st.header("Hello, World.")
def render(state):
    # Initialize Main Bar
    st.sidebar.title("Weight Watcher Menu")
    st.sidebar.image(icon_path)
    
    st.title("Weight Watcher Hub")
    st.header("Start living a healthier life today!")
    st.image(icon_path)
    
    cur_page = 'Login'
    for header in MENU:
        chosen = st.sidebar.button(header)
        if chosen:
            cur_page = header
    
    profile = state.__getattr__('profile')
    if profile is None:
        new_prof = render_not_logged_in()
        if new_prof is not None:
            state.__setattr__('profile', new_prof)
    else:
        new_prof = render_logged_in(profile)
        if new_prof is None:
            # They logged out
            state.__setattr__('profile', new_prof)
    return cur_page
    