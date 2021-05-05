import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./albums.csv')

data[data.isnull().any(axis=1)]

del data['mtv_critic']
del data['music_maniac_critic']

data = data.dropna()

after_rows = data.shape[0]
print(after_rows)
data

clean_data = data.copy()
clean_data['High_Score_Album'] = (clean_data['rolling_stone_critic'] > 3.0)*1
print(clean_data['High_Score_Album'])


y=clean_data[['High_Score_Album']].copy()
y

clean_data['High_Score_Album'].head()

features = ['num_of_sales']

X = clean_data[features].copy()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

rating_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
rating_classifier.fit(X_train, y_train)

predictions = rating_classifier.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)

