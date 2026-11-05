import csv

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X, y = [], []
with open("gestures.csv", newline="") as f:
    for row in csv.reader(f):
        if row:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
X, y = np.array(X), np.array(y)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))

model.fit(X, y)                                   # final model uses all the data
joblib.dump(model, "gesture_model.joblib")
print("saved gesture_model.joblib")