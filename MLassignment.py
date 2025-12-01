import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


#one-hot encode for categorical columns
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
