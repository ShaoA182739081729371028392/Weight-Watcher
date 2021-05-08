import pymongo
key =  'mongodb+srv://andrew:weightwatcher@cluster0.sin8q.mongodb.net/WeightWatcher?retryWrites=true&w=majority'
client = pymongo.MongoClient(key)
table = client['Weight Watcher']
class Profile:
    @classmethod
    def exists(cls, username):
        query = {'_id': username}
        found = table.find_one(query)
        for ex in found:
            if ex is not None:
                return True
        return None
    @classmethod
    def retrieve(cls, username, password):
        query = {'_id': username, 'password': password}
        found = table.find_one(query)
        for ex in found:
            if ex is not None:
                return ex
        return None
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