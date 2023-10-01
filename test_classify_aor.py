import pickle # we use this 1ib for dumping the model to binary forsat
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df =pd.read_csv('iris.data')

X = df.iloc[: ,:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)


classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)
print (score)

pickle_out = open("model_iris.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()