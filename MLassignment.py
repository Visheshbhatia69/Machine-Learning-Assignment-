import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

#loading and checking data
df = pd.read_csv("wallacecommunications.csv")
df = df.drop(['ID','town','country'],axis=1) #dropping unnecessary coloums
df['new_contract_this_campaign'] = df['new_contract_this_campaign'].replace({'yes':1,'no':0}) #converting target coloum

#defining categorical and numerical columns
cat_cols = ['job','married','education','arrears','housing','has_tv_package',
            'last_contact','conn_tr','last_contact_this_campaign_month',
            'outcome_previous_campaign']

num_cols = ['age','current_balance','last_contact_this_campaign_day',
            'this_campaign','days_since_last_contact_previous_campaign',
            'contacted_during_previous_campaign']

#one-hot encode categorical columns
df = pd.get_dummies(df,columns=cat_cols,drop_first=True)

#train/val/test split (60/20/20)
X = df.drop('new_contract_this_campaign',axis=1)
y = df['new_contract_this_campaign']
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.4,
                                                    random_state=42,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5,
                                                random_state=42,stratify=y_temp)

#scaling numerical columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

#applying smote
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train,y_train)
print("Class balance after SMOTE:")
print(y_train_sm.value_counts())

#logistic regression
lr = LogisticRegression(max_iter=1000)
lr_grid = GridSearchCV(lr,{'C':[0.01,0.1,1,10]},cv=5,scoring='f1')
lr_grid.fit(X_train_sm,y_train_sm)
