import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#models + tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE

#loading and checking data
df = pd.read_csv("wallacecommunications.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df['new_contract_this_campaign'].value_counts()) #checking imbalance
df = df.drop(['ID', 'town', 'country'], axis=1) #dropping unnecessary coloums
df['new_contract_this_campaign'] = df['new_contract_this_campaign'].replace({'yes':1,'no':0}) #converting target coloum

#defining categorical and numerical columns
cat_cols = ['job','married','education','arrears','housing','has_tv_package','last_contact','conn_tr','last_contact_this_campaign_month','outcome_previous_campaign']
num_cols = ['age','current_balance','last_contact_this_campaign_day','this_campaign','days_since_last_contact_previous_campaign','contacted_during_previous_campaign']

#one-hot encode categorical columns
df = pd.get_dummies(df,columns=cat_cols,drop_first=True)

#train/val/test split (60/20/20)
X = df.drop('new_contract_this_campaign',axis=1)
y = df['new_contract_this_campaign']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

#scaling numerical columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

#applying smote
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Class balance after SMOTE:")
print(y_train_sm.value_counts())

#logistic regression
lr = LogisticRegression(max_iter=1000)
lr_grid = GridSearchCV(lr, {'C':[0.01, 0.1, 1, 10]}, cv=5, scoring='f1')
lr_grid.fit(X_train_sm, y_train_sm)
lr_best = lr_grid.best_estimator_

#desicion tree
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt,{'max_depth':[5,10,20],'min_samples_split':[2,5,10]},cv=5,scoring='f1')
dt_grid.fit(X_train_sm, y_train_sm)
dt_best = dt_grid.best_estimator_

# XGBoost
xg = XGBClassifier(random_state=42, eval_metric='logloss')
xg_grid = GridSearchCV(xg,{'n_estimators':[50,100,200], 'learning_rate':[0.01,0.1,0.3]},cv=5,scoring='f1')
xg_grid.fit(X_train_sm, y_train_sm)
xgb_best = xg_grid.best_estimator_

#f1 scores
models = {
    'Logistic Regression': lr_best,
    'Decision Tree': dt_best,
    'XGBoost': xgb_best
}

f1_scores = []
for name, model in models.items():
    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred)
    f1_scores.append(f1)

plt.figure(figsize=(6, 4))
colors = ['skyblue', 'lightgreen', 'salmon']
plt.bar(models.keys(), f1_scores, color=colors)
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.title('Validation F1-Score Comparison')

for i, score in enumerate(f1_scores):
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center')

plt.show()

#final evaluatiom
print("\n===== FINAL TEST RESULTS =====")
for name, model in models.items():
    print("\n-----", name, "-----")
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name + " Confusion Matrix")
    plt.show()

#extra plottings
from xgboost import plot_importance
plot_importance(xgb_best, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.show()
