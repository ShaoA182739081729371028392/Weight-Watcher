'''
Features of the Application:
- BMI - True
- Goal Tracking
- graphs - Done
- Calorie Tracking
- Food Journal
- Exercise Journal 
- Health Tips
- Login for Privacy
'''
import pandas as pd
import streamlit as st
import datetime
import PIL.Image as Image
import sys
sys.path.append('..')
from back_end.profile import Profile
icon_path = './assets/logo.png'
MENU = [
    'Home', # Simply a Home page, with project details
    'Login',
    'Learn More'
]
def render_calorie_counting():
    # Performs the Calorie Counting Application, returns Calories Entered
    pass
def line():
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html = True)
    
def BMI():
    # Renders the BMI Calculate
    st.subheader("Check your progress! Compute your BMI using the following calculator.")
    default_weight = "Weight(In KG):"
    default_height = "Height(In M):"

    weight = st.text_input("", value = default_weight)
    height = st.text_input("", value = default_height)
    
    if weight != default_weight and height != default_height:
        try:
            weight = float(weight)
            height = float(height) ** 2
            st.write(f"Your BMI is currently: {weight / height}.")
        except:
            st.write("Invalid Inputs.")
def render_not_logged_in():
    st.header("Login or Register with Weight Watcher!")
    username = st.text_input("", value = "Username")
    password = st.text_input("", value = "Password")
    login = st.button("Login!")
    register = st.button("Register!")
    if login and (username != "" or password != '') and (username != 'Username' and password != 'Password'):
        # Check if the credentials are correct
        profile = Profile.retrieve(username, password)
        if profile is None:
            st.text("Credentials are Incorrect.")
        return profile
    else:
        if login:
            st.text("Invalid Credentials")
    if register and (username != "" or password != '') and (username != 'Username' and password != 'Password'):
        # Check if the credentials already exist
        exists = Profile.exists(username)
        if exists:
            st.text("Username already exists.")
        else:
            Profile.insert(username, password)
            return Profile.retrieve(username, password)
    else:
        if register:
            st.text("Invalid Credentials")
    return None
def format_date(year, month, day):
    date = datetime.datetime(year, month, day)
    string = date.strftime("%d %B, %Y")
    return string
def prune(year, month, day, to_prune):
    for last_year, last_month, last_day in to_prune:
        if expired(year, month, day, last_year, last_month, last_day):
            del to_prune[last_year, last_month, last_day]
def expired(year, month, day, last_year, last_month, last_day):
    last_date = datetime.date(last_year, last_month, last_day)
    cur_date = datetime.date(year, month, day)
    time_diff = cur_date - last_date
    return time_diff.day > 7 
def render_logged_in(profile):
    # Extract Current Time to Update Profiles and Graph 
    cur_date = datetime.datetime.now()
    year = cur_date.year
    month = cur_date.month
    day = cur_date.day
    hour = cur_date.hour

    # Extract Data from Profile
    username = profile['_id']
    calorie_totals = profile['calorie_totals']
    exercise_totals = profile['exercise_totals']
    calorie_goal = profile['calorie_goal']
    exercise_goal = profile['exercise_goal']
    calorie_journal = profile['calorie_journal']
    exercise_journal = profile['exercise_journal']
    last_year, last_month, last_day, last_hour = profile['last_updated']
    # Prune Entries
    prune(year, month, day, calorie_totals)
    prune(year, month, day, exercise_totals)
    prune(year, month, day, calorie_journal)
    prune(year, month, day, exercise_journal)
    # Display Data to the user
    st.header(f"Hello, {username}!")
    st.write("Here\'s your weekly summary.")
    # Calorie Counting Application:
    line()
    calories, food_type, weight, volume = render_calorie_counting()
    # Convert the Calorie Totals to a Line Graph
    line()
    df_to_be = {'index': [], 'values': []}
    for y, m, d in calorie_totals:
        df_to_be['index'] += [format_date(y, m, d)]
        df_to_be['values'] += [calorie_totals[(y, m, d)]]
    df = pd.DataFrame(df_to_be)
    df = df.set_index('index')
    st.subheader("This Week\'s Calorie Intake:")
    st.line_chart(data = df)
    if len(df) == 0:
        st.write("No Data Present Yet.")
    
    line()

    st.subheader("This Week\'s Minutes of Exercise: ")
    # Render Exercise
    df_to_be = {'index': [], 'values': []}
    for y, m, d in exercise_journal:
        df_to_be['index'] += [format_date(y, m, d)]
        df_to_be['values'] += [calorie_totals[(y, m, d)]]
    df = pd.DataFrame(df_to_be)
    df = df.set_index('index')

    st.line_chart(data = df)
    if len(df) == 0:
        st.write('No Data Present Yet.')
    # Render BMI
    line()
    BMI()
    line()
    # Render 
    return profile
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
    