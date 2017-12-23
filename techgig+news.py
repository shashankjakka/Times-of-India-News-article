
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from vecstack import stacking
from keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier
from keras.layers import Dense,Dropout,Activation
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,roc_auc_score
pd.set_option("max_columns",50)
pd.set_option("max_columns",50)


# In[ ]:


## scikit wrapper for native xgboost
class XGBoostClassifier():
    def __init__(self,num_boost_round=10,**params):
        self.clf=None
        self.num_boost_round=num_boost_round
        self.classes_=[0,1]
        self.params=params
        self.params.update({'objective':"multi:softprob"}) 
    def classes_(self):
        l=[0,1]
        return l
    def fit(self,X,y,num_boost_round=None):
        num_boost_round=num_boost_round or self.num_boost_round
        self.label2num=dict((label,i) for i,label in enumerate(sorted(set(y))))
        dtrain=xgb.DMatrix(X,label=[self.label2num[label] for label in y])
        self.clf=xgb.train(params=self.params,dtrain=dtrain,num_boost_round=num_boost_round)
    def predict(self,X):
        num2label=dict((i,label)for label,i in self.label2num.items())
        Y=self.predict_proba(X)
        y=np.argmax(Y,axis=1)
        return np.array([num2label[i] for i in y])
    def predict_proba(self,X):
        dtest=xgb.DMatrix(X)
        return self.clf.predict(dtest)
    def score(self,X,y):
        Y=self.predict_proba(X)
        return 1/logless(y,Y)
    def get_params(self,deep=True):
        return self.params
    def set_params(self,**params):
        if 'num_boost_round' in params:
            self.num_boost_round=params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
                                            


# In[ ]:


temp1=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_1.csv')
temp2=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_2.csv')
temp3=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_3.csv')
temp4=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_4.csv')
temp5=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_5.csv')
temp6=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_6.csv')
temp7=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_7.csv')
temp8=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_8.csv')
temp9=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_9.csv')
temp10=pd.read_csv('/Users/shashank/Downloads/News/Train/Train_10.csv')
train=pd.concat([temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10])


# In[ ]:


temp1=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_1.csv')
temp2=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_2.csv')
temp3=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_3.csv')
temp4=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_4.csv')
temp5=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_5.csv')
temp6=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_6.csv')
temp7=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_7.csv')
temp8=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_8.csv')
temp9=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_9.csv')
temp10=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_10.csv')
temp11=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_11.csv')
temp12=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_12.csv')
temp13=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_13.csv')
temp14=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_14.csv')
temp15=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_15.csv')
temp16=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_16.csv')
temp17=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_17.csv')
temp18=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_18.csv')
temp19=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_19.csv')
temp20=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_20.csv')
temp21=pd.read_csv('/Users/shashank/Downloads/News/impression_attr/impression_attr_21.csv')
imp=pd.concat([temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11,temp12,temp13,temp14,
              temp15,temp16,temp17,temp18,temp19,temp20,temp21])


# In[ ]:


users=pd.read_csv("/Users/shashank/Downloads/News/user_item.csv")


# In[ ]:


temp1=pd.read_excel("/Users/shashank/Downloads/News/Item_Category_Map/Item_Category_Map_1.xlsx")
temp2=pd.read_excel("/Users/shashank/Downloads/News/Item_Category_Map/Item_Category_Map_2.xlsx")
categories=pd.concat([temp1,temp2])


# In[ ]:


temp1=pd.read_csv("/Users/shashank/Downloads/News/Item_Attributes/Item_Attributes_1.csv")
temp2=pd.read_csv("/Users/shashank/Downloads/News/Item_Attributes/Item_Attributes_2.csv")
temp3=pd.read_csv("/Users/shashank/Downloads/News/Item_Attributes/Item_Attributes_3.csv")
attributes=pd.concat([temp1,temp2,temp3])


# In[ ]:


test=pd.read_csv("/Users/shashank/Downloads/News/test-data.csv")


# In[ ]:


## drop the duplicates
temp=train[train.duplicated(['impression_id','item_id'],keep=False)]
temp.drop_duplicates(['impression_id','item_id'],inplace=True)
temp["click"]=2
train.drop_duplicates(['impression_id','item_id'],keep=False,inplace=True)
train=pd.concat([train,temp])


# In[ ]:


le=LabelEncoder()
imp["refrenceUrl"]=le.fit_transform(imp["refrenceUrl"])

le=LabelEncoder()
imp["uuid"]=le.fit_transform(imp["uuid"])

imp["timestamp_impression"]=pd.to_datetime(imp["timestamp_impression"])
imp['timestamp_impression']=imp['timestamp_impression'].apply(lambda x: datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S.%f'))
imp['Day']=imp['timestamp_impression'].apply(lambda x:x.day)
imp['hour']=imp['timestamp_impression'].apply(lambda x:x.hour)
imp['minute']=imp['timestamp_impression'].apply(lambda x:x.minute)


# In[ ]:


# merge train and imp on impression_id
train=pd.merge(train,imp,on="impression_id",how="inner")


# In[ ]:


## merge 
test=pd.merge(test,imp,on="impression_id",how="inner")


# In[ ]:


temp=train.groupby(["item_id","click"],as_index=False).count()[["item_id","click","impression_id"]]
temp2=temp.groupby(["item_id"],as_index=False).sum()[["item_id","impression_id"]]
temp2.columns=["item_id","count"]
temp=pd.merge(temp,temp2,on="item_id",how="inner")
temp["per_item"]=temp["impression_id"]/temp["count"]
temp=temp[temp["click"]==2]
train=pd.merge(train,temp[["item_id","per_item"]],on="item_id",how="left")
test=pd.merge(test,temp[["item_id","per_item"]],on="item_id",how="left")


# In[ ]:


temp=user.groupby(["uuid"],as_index=False).count()
temp.columns=["uuid","count"]
train=pd.merge(train,temp[["uuid","count"]],on="uuid",how="left")
test=pd.merge(test,temp[["uuid","count"]],on="uuid",how="left")


# In[ ]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
Y=train.loc[:,"click"]
train.drop(["click"],axis=1,inplace=True)


# In[ ]:


train.drop(["impression_id","refrenceUrl","timestamp_impression",
            "uvh"],axis=1,inplace=True)


# In[ ]:


test.drop(["impression_id","refrenceUrl","timestamp_impression",
            "uvh"],axis=1,inplace=True)


# In[ ]:


train['target']=Y
predictors=[x for x in train.columns if x not in ['target']]
target=['target']


# In[ ]:


## training starts


# In[ ]:


kfold=KFold(n_splits=3)


# In[ ]:


clf=RandomForestClassifier(n_estimators=250)
clf1=LogisticRegression()


# In[ ]:


scores=cross_val_score(estimator=clf,train[predictors].values,train[target].values.ravel(),cv=kfold,scoring='neg_mean_squared_error')
print scores.mean()


# In[ ]:


scores=cross_val_score(estimator=clf1,train[predictors].values,train[target].values.ravel(),cv=kfold,scoring='neg_mean_squared_error')
print scores.mean()


# In[ ]:


ratio=float(np.sum(Y==1)/float(np.sum(Y==0)))

params={
    'objective':'binary:logistic',
    'scale_pos_weight':ratio,
    'eta':0.2
}


# In[ ]:


scores=cross_val_score(estimator=XGBoostClassifier(num_class=2,num_boost_round=100,params),
                                                   ,train[predictors].values,train[target].values.ravel()
                                                       ,cv=kfold,scoring='neg_mean_squared_error')
print scores.mean()


# In[ ]:


## HyperTune xgb model


# In[ ]:


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=2, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# In[ ]:


## Fix learning rate =0.1


# In[2]:


## get n_estimators

ratio=float(np.sum(Y==1)/float(np.sum(Y==0)))
params={
    'objective':'binary:logistic',
    'eta':0.1,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'scale_pos_weight':ratio,
    'eta':0.2
}

xgb1 = =XGBoostClassifier(num_class=2,num_boost=1000,params)

modelfit(xgb1, train, predictors)


# In[ ]:


## n_estimators=119


# In[ ]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBoostClassifier(num_class=2,num_boost_round=119,params), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False,verbose=2,cv=2)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


## best 'max_depth': 3, 'min_child_weight': 1 check further


# In[ ]:


param_test2 = {
 'min_child_weight':[2,4],
    'max_depth':[1,2],
}
gsearch2 = GridSearchCV(estimator = XGBoostClassifier(num_class=2,num_boost_round=119,params), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=2,verbose=2)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


### 'max_depth': 2, 'min_child_weight': 1 are ideal change them


# In[ ]:


params['max_depth']=2 . ## expected as data is imbalanced
params['min_child_weight']=1


# In[ ]:


## tune gamma


# In[ ]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBoostClassifier(num_class=2,num_boost_round=119,params),
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=2,verbose=2)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


#gamma=0.1,update gamma


# In[ ]:


params['gamma']=0.1


# In[ ]:


# cross check for num_estimators again with updated optimal parameters


# In[ ]:


modelfit(xgb1, train, predictors)


# In[ ]:


## update num_boost to 148


# In[ ]:


## tune subsample and colsample_bytree


# In[ ]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBoostClassifier(num_class=2,num_boost_round=148,params), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=2,verbose=2)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


## colsample_bytree-0.8,subsample:0.9


# In[ ]:


params['colsample_bytree']=0.8
params['subsample']=0.9


# In[ ]:


## tune reg_aplha


# In[ ]:


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBoostClassifier(num_class=2,num_boost_round=148,params), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=2,verbose=2)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[ ]:


##  'reg_alpha': 1e-05 ,maybe further tune? No
## check n_estimators again with updates params


# In[ ]:


params['reg_alpha']=1e-05


# In[ ]:


modelfit(xgb1, train, predictors)


# In[ ]:


## n_estimators=148 same!


# In[ ]:


## final params are 'eta':0.1,'max_depth':2,'min_child_weight':1,'gamma':0.1,'subsample':0.9,
##    'colsample_bytree':0.8,'reg_alpha':1e-05 n_estimators:148


# In[ ]:


def base_model():
    model = Sequential()
    model.add(Dense(1000, input_dim=train[predictors].shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1000, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(500, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dropout(0.15))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


NN = KerasClassifier(build_fn=base_model, nb_epoch=25, batch_size=64, verbose=2)


# In[ ]:


scores=cross_val_score(estimator=NN,train[predictors].values,train[target].values,cv=kfold,scoring='neg_mean_squared_error')
print scores.mean()


# In[ ]:


## stack,only xgb is tuned,maybe tune others sometime?

models = [
    ExtraTreesClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 250,),
        
    RandomForestClassifier(random_state = 0, n_jobs = -1, 
        n_estimators = 250),
    XGBoostClassifier(num_class=2,num_boost_round=148,params)
    
    KerasClassifier(build_fn=base_model, nb_epoch=25, batch_size=64, verbose=2)
]
    
## stack KNN too? taking too long,leave it!


# In[ ]:


S_train, S_test = stacking(models, train[predictors].values, Y[target].values.ravel(), test[predictors].values, 
    regression = False, metric = accuracy_score, n_folds = 3, 
    shuffle = True, random_state = 0, verbose = 2)


# In[ ]:


## Add stacked features to train and test
train['ExtraTree']=S_train[:,0]
train['RandomForest']=S_train[:,1]
train['Xgb']=S_train[:,2]
train['Keras']=S_train[:,3]


# In[ ]:


test['ExtraTree']=S_test[:,0]
test['RandomForest']=S_test[:,1]
test['Xgb']=S_test[:,2]
test['Keras']=S_test[:,3]


# In[ ]:


## End of first level


# In[ ]:


## Second Level start ( Xgb and base NN and extratree bagging is used)


# In[ ]:


## Find optimal weight for bagging(use both geometric and arthematic progression and ) 
## ap1 * [XGBOOST^gp1 * NN^gp2] + ap2 * [ET]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train[predictors].values,train[target].values.ravel(),test_size=0.3)


# In[ ]:


xgb=XGBoostClassifier(num_class=2,num_boost_round=148,params)
xgb.fit(X_train,y_train)
probs1=xgb.predict_proba(X_test)
probs1=probs1[:,1]
XGBOOST=(probs1>0.4).astype('int')


# In[ ]:


nn = KerasClassifier(build_fn=base_model, nb_epoch=25, batch_size=64, verbose=2) ## tune the model? No,taking too long.
nn.fit(X_train,y_train)
probs2=nn.predict_proba(X_test)
probs2=probs2[:,1]
NN=(probs2>0.4).astype('int')


# In[ ]:


et=ExtraTreesClassifier(n_estimators=250)
et.fit(X_train,y_train)
probs1=et.predict_proba(X_test)
probs1=probs1[:,1]
ET=(probs1>0.4).astype('int')


# In[ ]:


## find optimal weights for bagging
ap1=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,1]
ap2=[0.5,0.4,0.3,0.2,0.15,0.1,0.05,0]
gp1=[0.5,0.6,0.7,0.8,0.85,0.9,0.95,1]
gp2=[0.5,0.4,0.3,0.2,0.15,0.1,0.05,0]


# In[ ]:


for i,j in zip(ap1,ap2):
    for p,q in zip(gp1,gp2):
        final=i * [np.power(XGBOOST,p) * np.power(NN,q)] + j * [ET]
        print mean_squared_error(y_test,final)


# In[ ]:


## best weights are 0.90 * [XGBOOST^0.7 * NN^0.3] + 0.1 * [ET]


# In[ ]:


xgb=XGBoostClassifier(num_class=2,num_boost_round=148,params)
xgb.fit(train[predictors].values,train[target].values.ravel())
probs1=xgb.predict_proba(test[predictors].values)
probs1=probs1[:,1]
XGBOOST=(probs1>0.5).astype('int') ## maybe calibrate? Calibration scored low!! what could be the reason?


# In[ ]:


nn = KerasClassifier(build_fn=base_model, nb_epoch=25, batch_size=64, verbose=2) ## tune the model? No,taking too long.


# In[ ]:


nn.fit(train[predictors].values,train[target].values.ravel())
probs2=nn.predict_proba(test[predictors].values)
probs2=probs2[:,1]
NN=(probs2>0.5).astype('int')


# In[ ]:


et=ExtraTreesClassifier(n_estimators=250)
et.fit(train[predictors].values,train[target].values.ravel())
probs2=et.predict_proba(test[predictors].values)
probs2=probs2[:,1]
ET=(probs2>0.5).astype('int')


# In[ ]:


## Bag the models with tuned weights


# In[ ]:


final= 0.9* [np.power(XGBOOST,0.7) * np.power(NN,0.3)] + 0.1 * [ET]

