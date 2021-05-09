'''
Features of the Application:
- BMI - True
- Goal Tracking
- graphs - Done
- Calorie Tracking - Done
- Food Journal - Done
- Exercise Journal  - Done
- Login for Privacy
'''
import pandas as pd
import streamlit as st
import datetime
import PIL.Image as Image
import sys
sys.path.append('..')
from back_end.profile import Profile
from deep_learning import calorie_counter
icon_path = './assets/logo.png'
source_code_link = 'https://github.com/ShaoA182739081729371028392/Weight-Watcher'

st.set_option('deprecation.showfileUploaderEncoding', False)
sample_images = 'https://github.com/ShaoA182739081729371028392/Weight-Watcher/tree/main/sample%20images'
MENU = [
    'Home', # Simply a Home page, with project details
    'Login',
    'Learn More'
]
def set_calorie_goals(calorie_goal):
    st.subheader("Set a Calorie Goal.")
    goal = st.text_input('', value = "Calorie Goal: ")
    entered = st.button("Set/Create!", key ='sniasc')
    if entered:
        try:
            calorie_goal = eval(goal)
            return calorie_goal
        except:
            return calorie_goal
    return calorie_goal
def render_calorie_goals(calorie_totals, calorie_goal):
    cur_date = datetime.datetime.now()
    year = cur_date.year
    month = cur_date.month
    day = cur_date.day
    st.subheader("Current Calorie Goals:")
    cur_calories = calorie_totals[(year, month, day)]
    if calorie_goal is not None:
        calories_left = calorie_goal - cur_calories 
        st.write(f"You have {calories_left} calories left to consume in the day before reaching your goal!")
    else:
        st.write("Set a Goal!")
    
def set_exercise_goals(exercise_goal):
    st.subheader("Enter an Exercise Goal!")
    goal = st.text_input('', value = "Exercise Goal: ")
    entered = st.button("Set/Create!", key = 'ewefwfwef')
    if entered:
        try:
            exercise_goal = eval(goal)
            return exercise_goal
        except:
            return exercise_goal
    return exercise_goal

def render_exercise_goals(exercise_journal, exercise_goal):
    cur_date = datetime.datetime.now()
    year = cur_date.year
    month = cur_date.month
    day = cur_date.day
    cur_exercise = exercise_journal[(year, month, day)]
    st.subheader("Current Exercise Goals.")
    if exercise_goal:
        exercise_to_go = exercise_goal - cur_exercise
        st.write(f'You Have {exercise_to_go} minutes left of exercise to go!')
    
def convert_to_dict(dictionary):
    for string in list(dictionary.keys()):
        if type(string) == type('---'):
            dictionary[eval(string)] = dictionary[string]
            del dictionary[string]
def convert_to_string(dictionary):
    # MongoDB can only handle strings, so convert every entry
    for entry in list(dictionary.keys()):
        dictionary[str(entry)] = dictionary[entry]
        del dictionary[entry]
def render_calorie_counting():
    # Performs the Calorie Counting Application, returns Calories Entered
    st.subheader("Count Calories using Weight Watcher! Snap a Pic and then Eat it!")
    st.write("Calorie Counting is performed using a Segmentation-Pretrained EfficientNetB0 to perform Multi-Task Learning on Food Classification, Weight, and Volume Estimation. The Model achieves 98 percent F1 Score and 90 percent accuracy, with 10g average error on weight.")
    st.write(f"Images should be taken with a 1 Yuan Coin in the Background, as this provides the neural network with a scale of how large the image(and food) is, so it can perform proper estimation of the weight(Otherwise, weight estimation is impossible). Models were trained in 8 hours during TOHacks 2021, on the ECUSTFD dataset. Sample Images can be found [here]({sample_images}). Feel Free to download and test them!")
    files = st.file_uploader("Count Your Calories!", type = ['png', 'jpg', 'jpeg'])
    if files is not None:
        image = Image.open(files)
        st.image(image)
        class_name, weight, volume, calories = calorie_counter.process_image(image)
        weight = round(weight, 3)
        volume = round(volume, 3)
        calories = round(calories, 3)
        return class_name, weight, volume, calories
    return None, None, None, None
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
def prune_journal(year, month, day, to_prune):
    for last_year, last_month, last_day, last_hour, last_minute in to_prune:
        if expired(year, month, day, last_year, last_month, last_day):
            del to_prune[last_year, last_month, last_day, last_hour, last_minute]
def prune_totals(year, month, day, to_prune):
    for last_year, last_month, last_day in to_prune:
        if expired(year, month, day, last_year, last_month, last_day):
            del to_prune[last_year, last_month, last_day]
def expired(year, month, day, last_year, last_month, last_day):
    last_date = datetime.date(last_year, last_month, last_day)
    cur_date = datetime.date(year, month, day)
    time_diff = cur_date - last_date
    return time_diff.days > 7 
def initialize_totals(year, month, day, entry):
    if (year, month, day) not in entry:
        entry[(year, month, day)] = 0
def initialize_journals(year, month, day, hour, minute, journal):
    if journal is None:
        return {}
    return journal
def render_exercise():
    # Renders Exercise Input
    st.subheader("Get outside and exercise :) Enter your exercise here.")
    exercise_type = st.text_input("", value = 'Exercise Type')
    minutes = st.text_input("", value = 'Minutes Exercised')
    enter = st.button("Enter!")
    if enter:
        try:
            minutes = int(minutes)
            return exercise_type, minutes
        except:
            return None, None
    return None, None
def render_calorie_journal(journal):
    st.header("Weekly Calorie Journal.")
    entries = []
    for year, month, day, hour, minute in journal:
        date = datetime.datetime(year, month, day, hour, minute)
        date = date.strftime("%Y %b %d, %H: %M")
        date = f"Time: {date}, Ate: {journal[(year, month, day, hour, minute)][0]}, Calories: {journal[(year, month, day, hour, minute)][1]}"
        entries += [date]
    st.multiselect("", entries)
def render_exercise_journal(journal):
    st.header("Weekly Exercise Journal.")
    entries = []
    for year, month, day, hour, minute in journal:
        date = datetime.datetime(year, month, day, hour, minute)
        date = date.strftime("%Y %b %d, %H: %M")
        date = f"Time: {date}, Did: {journal[(year, month, day, hour, minute)][0]}, Minutes: {journal[(year, month, day, hour, minute)][1]}"
        entries += [date]
    st.multiselect("", entries)
def render_logged_in(profile):
    # Extract Current Time to Update Profiles and Graph 
    cur_date = datetime.datetime.now()
    year = cur_date.year
    month = cur_date.month
    day = cur_date.day
    hour = cur_date.hour
    minute = cur_date.minute

    # Extract Data from Profile
    username = profile['_id']
    password = profile['password']
    calorie_totals = profile['calorie_totals']
    exercise_totals = profile['exercise_totals']
    calorie_goal = profile['calorie_goal']
    exercise_goal = profile['exercise_goal']
    calorie_journal = profile['calorie_journal']
    exercise_journal = profile['exercise_journal']
    # Convert to Tuples
    to_change = [calorie_totals, exercise_totals, calorie_journal, exercise_journal]
    for ex in to_change:
        convert_to_dict(ex)
    initialize_totals(year, month, day, calorie_totals)
    initialize_totals(year, month, day, exercise_totals)
    calorie_journal = initialize_journals(year, month, day, hour, minute, calorie_journal)
    exercise_journal = initialize_journals(year, month, day, hour, minute, exercise_journal)

    # Prune Entries
    prune_totals(year, month, day, calorie_totals)
    prune_totals(year, month, day, exercise_totals)
    prune_journal(year, month, day, calorie_journal)
    prune_journal(year, month, day, exercise_journal)
    # Display Data to the user
    line()
    st.header(f"Hello, {username}!")
    # Calorie Counting Application:
    line()
    class_name, weight, volume, calories = render_calorie_counting()
    if class_name is not None:
        st.write(f"Predicted Food: {class_name}")
        st.write(f"Weighs: {weight}g, Volume: {volume}cm^3, calories: {calories}kCal.")
        add = st.button("Add Calories to Consumed?")
        if add:
            calorie_totals[(year, month, day)] += calories
            calorie_journal[(year, month, day, hour, minute)] = (class_name, calories)
            st.write("Calories Added. Graph Updated.")
    # Calorie Goals
    line()
    calorie_goal = set_calorie_goals(calorie_goal)
    render_calorie_goals(calorie_totals, calorie_goal)
    # Calorie Journal
    line()
    render_calorie_journal(calorie_journal)
    # Render Exercise Input
    line()
    exercise_type, minutes = render_exercise()
    if exercise_type is not None:
        exercise_totals[(year, month, day)] += minutes
        exercise_journal[(year, month, day, hour, minute)] = (exercise_type, minutes)
    # Exercise Goals:
    line()
    exercise_goal = set_exercise_goals(exercise_goal)
    print(exercise_goal)
    render_exercise_goals(exercise_totals, exercise_goal)

    # Exercise Journal
    line()
    render_exercise_journal(exercise_journal)
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
    for y, m, d in exercise_totals:
        df_to_be['index'] += [format_date(y, m, d)]
        df_to_be['values'] += [exercise_totals[(y, m, d)]]
    df = pd.DataFrame(df_to_be)       
    df = df.set_index('index')

    st.line_chart(data = df)
    if len(df) == 0:
        st.write('No Data Present Yet.')
    # Render BMI
    line()
    BMI()
    line()
    # Update MongoDB with all entries
    # Change all dicts to strings 
    to_change = [calorie_totals, exercise_totals, calorie_journal, exercise_journal]
    for ex in to_change:
        convert_to_string(ex)
    Profile.update(username, password, calorie_totals = calorie_totals, exercise_totals = exercise_totals, calorie_goal = calorie_goal, exercise_goal = exercise_goal, calorie_journal = calorie_journal, exercise_journal = exercise_journal)
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
    st.sidebar.info(f"All Source Code can be found [here]({source_code_link})")
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
    