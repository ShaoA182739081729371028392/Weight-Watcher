import pymongo
key =  'mongodb+srv://andrew:weightwatcher@cluster0.sin8q.mongodb.net/WeightWatcher?retryWrites=true&w=majority'
client = pymongo.MongoClient(key)
cluster = client['WeightWatcher']
table = cluster['Profiles']
import datetime 
class Profile:
    @classmethod
    def exists(cls, username):
        query = {'_id': username}
        found = table.find_one(query)
        if found is None:
            return None
        return True
    @classmethod
    def retrieve(cls, username, password):
        query = {'_id': username, 'password': password}
        found = table.find_one(query)
        if found is None:
            return None
        return found
    @classmethod
    def update(cls, username, calorie_totals = None, exercise_totals = None, calorie_goal = None, exercise_goal = None, calorie_journal = None, exercise_journal = None):
        query = {'_id': username}
        replace = {}
        if calorie_totals is not None:
            replace['calorie_totals'] = calorie_totals
        if exercise_totals is not None:
            replace['exercise_totals'] = exercise_totals
        if calorie_goal is not None:
            replace['calorie_goal'] = calorie_goal
        if exercise_goal is not None:
            replace['exercise_goal'] = exercise_goal
        if calorie_journal is not None:
            replace['calorie_journal'] = calorie_journal 
        if exercise_journal is not None:
            replace['exercise_journal'] = exercise_journal
        print(query)
        print(replace)
        table.replace_one(query, replace)
    @classmethod
    def insert(cls, username, password, calorie_totals = {}, exercise_totals = {}, calorie_goal = None, exercise_goal = None, calorie_journal = {}, exercise_journal = {}):
        query = {
            '_id': username, 
            'password': password,
            'calorie_totals': calorie_totals,
            'exercise_totals': exercise_totals,
            'calorie_goal': calorie_goal,
            'exercise_goal': exercise_goal,
            'calorie_journal': calorie_journal,
            'exercise_journal': exercise_journal
        }
        table.insert_one(query)