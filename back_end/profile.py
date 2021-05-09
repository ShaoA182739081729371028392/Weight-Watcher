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

        query = {'_id': username}#, 'password': password}
        found = table.find_one(query)
        print(found)
        if found is None:
            return None
        return found
    @classmethod
    def update(cls, username, password, calorie_totals = None, exercise_totals = None, calorie_goal = None, exercise_goal = None, calorie_journal = None, exercise_journal = None):
        query = {'_id': username, 'password': password}
        replace = {}
        replace['password'] = password
        replace['calorie_totals'] = calorie_totals
        replace['exercise_totals'] = exercise_totals
        replace['calorie_goal'] = calorie_goal
        replace['exercise_goal'] = exercise_goal
        replace['calorie_journal'] = calorie_journal 
        replace['exercise_journal'] = exercise_journal
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