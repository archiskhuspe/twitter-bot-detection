"""Train the Twitter bot/human classifier and persist it to model.pkl.

Feature engineering: keyword/lexicon binary flags over the account's
screen name, name, description and status, plus numeric profile signals.
A Decision Tree (entropy) is trained on the full dataset and pickled for
the Flask app (app.py) to load.
"""

import pickle

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

training_data = pd.read_csv("data/training_data_2_csv_UTF.csv")

# feature engineering
bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                   r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                   r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                   r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)

# feature extraction
training_data['listed_count_binary'] = (training_data.listed_count > 20000) == False
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count',
            'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

# implementing classifiers
inputs = training_data[features].iloc[:, :-1]
target = training_data[features].iloc[:, -1]

le_screen_name_binary = LabelEncoder()
le_name_binary = LabelEncoder()
le_description_binary = LabelEncoder()
le_status_binary = LabelEncoder()
le_verified = LabelEncoder()
le_followers_count = LabelEncoder()
le_friends_count = LabelEncoder()
le_statuses_count = LabelEncoder()
le_listed_count_binary = LabelEncoder()

inputs['screen_name_binary_n'] = le_screen_name_binary.fit_transform(inputs['screen_name_binary'])
inputs['name_binary_n'] = le_name_binary.fit_transform(inputs['name_binary'])
inputs['description_binary_n'] = le_description_binary.fit_transform(inputs['description_binary'])
inputs['status_binary_n'] = le_status_binary.fit_transform(inputs['status_binary'])
inputs['verified_n'] = le_verified.fit_transform(inputs['verified'])
inputs['followers_count_n'] = le_followers_count.fit_transform(inputs['followers_count'])
inputs['friends_count_n'] = le_friends_count.fit_transform(inputs['friends_count'])
inputs['statuses_count_n'] = le_statuses_count.fit_transform(inputs['statuses_count'])
inputs['listed_count_binary_n'] = le_listed_count_binary.fit_transform(inputs['listed_count_binary'])

inputs_n = inputs.drop(
    ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count',
     'friends_count', 'statuses_count', 'listed_count_binary'], axis='columns')

dt = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)
dt.fit(inputs_n, target)

print(dt.score(inputs_n, target))

pickle.dump(dt, open('model.pkl', 'wb'))
