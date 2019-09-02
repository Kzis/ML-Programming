import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

df = pd.read_csv('train.csv', nrows=1e5)
print('dataset loaded')

device_type = pd.get_dummies(df.device_type, drop_first=True)
device_conn_type = pd.get_dummies(df.device_conn_type, drop_first=True)
app_category = pd.get_dummies(df.app_category, drop_first=True)

X = df[['banner_pos', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']]
X = pd.concat([X, device_type, device_conn_type, app_category], axis=1)

y = df['click']

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.8,
                                                    random_state=112)
logreg = LogisticRegression(solver='liblinear',
                            max_iter=1000,
                            verbose=100)
logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(X_train, y_train)

print("Best Parameter: {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))

logreg = LogisticRegression(solver='liblinear',
                            C=logreg_cv.best_params_['C'],
                            penalty= logreg_cv.best_params_['penalty'])

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

