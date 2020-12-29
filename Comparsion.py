import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

heart_df = pd.read_csv('./data/heart.csv')
print(heart_df)

fig, axes = plt.subplots(1,2,figsize=(20, 8))
ax = heart_df.target.value_counts().plot(kind="bar", ax=axes[0])


ax.set_title("患者分布", fontsize=20)
ax.set_xlabel("1：患病，0：未患病", fontsize=20)

heart_df.target.value_counts().plot(kind="pie", autopct="%.2f%%", labels=['患病', '未患病'], ax=axes[1], fontsize=15)
# plt.savefig("./aa.png")
plt.show()

# 性别和患病的分布
ax1 = plt.subplot(121)
ax = sns.countplot(x="sex",hue='target',data=heart_df,ax=ax1)
ax.set_xlabel("0：女性，1：男性", fontsize=20)

ax2 = plt.subplot(222)
heart_df[heart_df['target'] == 0].sex.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['男性','女性'],ax=ax2, fontsize=15)
ax2.set_title("未患病性别比例", fontsize=20)

ax2 = plt.subplot(224)
heart_df[heart_df['target'] == 1].sex.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['男性','女性'],ax=ax2, fontsize=15)
ax2.set_title("患病性别比例", fontsize=20)

plt.show()

fig,axes = plt.subplots(2,1,figsize=(20,10))
sns.countplot(x="age",hue="target",data=heart_df,ax=axes[0])

# 0-45：青年人，45-59：中年人，60-100：老年人
age_type = pd.cut(heart_df.age,bins=[0,45,60,100],include_lowest=True,right=False,labels=['青年人','中年人','老年人'])
age_target_df = pd.concat([age_type,heart_df.target],axis=1)
sns.countplot(x="age",hue='target',data=age_target_df)
plt.show()

plt.figure(figsize=(20,8),dpi=80)
sns.heatmap(heart_df.corr(),cmap="Blues",annot=True)
plt.show()


# 数据预处理
features = heart_df.drop(columns=['target'])
targets = heart_df['target']

# 将离散型数据，从普通的0,1,2这些，转换成真正的字符串表示

# sex
features.loc[features['sex']==0,'sex'] = 'female'
features.loc[features['sex']==1,'sex'] = 'male'
# cp
features.loc[features['cp'] == 1,'cp'] = 'typical'
features.loc[features['cp'] == 2,'cp'] = 'atypical'
features.loc[features['cp'] == 3,'cp'] = 'non-anginal'
features.loc[features['cp'] == 4,'cp'] = 'asymptomatic'
# fbs
features.loc[features['fbs'] == 1,'fbs'] = 'true'
features.loc[features['fbs'] == 0,'fbs'] = 'false'
# exang
features.loc[features['exang'] == 1,'exang'] = 'true'
features.loc[features['exang'] == 0,'exang'] = 'false'
# slope
features.loc[features['slope'] == 1,'slope'] = 'true'
features.loc[features['slope'] == 2,'slope'] = 'true'
features.loc[features['slope'] == 3,'slope'] = 'true'
# thal
features.loc[features['thal'] == 3,'thal'] = 'normal'
features.loc[features['thal'] == 3,'thal'] = 'fixed'
features.loc[features['thal'] == 3,'thal'] = 'reversable'
# restecg
# 0：普通，1：ST-T波异常，2：可能左心室肥大
features.loc[features['restecg'] == 0,'restecg'] = 'normal'
features.loc[features['restecg'] == 1,'restecg'] = 'ST-T abnormal'
features.loc[features['restecg'] == 2,'restecg'] = 'Left ventricular hypertrophy'
# ca
features['ca'].astype("object")
# thal
features.thal.astype("object")

print(features)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

features = pd.get_dummies(features)
features_temp = StandardScaler().fit_transform(features)

X_train,X_test,y_train,y_test = train_test_split(features_temp, targets, test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score,auc


def plotting(estimator,y_test,y_name):
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    y_predict_proba = estimator.predict_proba(X_test)
    precisions,recalls,thretholds = precision_recall_curve(y_test,y_predict_proba[:,1])
    axes[0].plot(precisions,recalls)
    axes[0].set_title("%s 平均精准率：%.2f" % (y_name, average_precision_score(y_test,y_predict_proba[:,1])), fontsize=20)
    axes[0].set_xlabel("召回率", fontsize=20)
    axes[0].set_ylabel("精准率", fontsize=20)

    fpr,tpr,thretholds = roc_curve(y_test,y_predict_proba[:,1])
    axes[1].plot(fpr,tpr)
    axes[1].set_title("%s AUC值：%.2f"%(y_name, auc(fpr,tpr)), fontsize=20)
    axes[1].set_xlabel("FPR", fontsize=20)
    axes[1].set_ylabel("TPR", fontsize=20)
    plt.savefig("./imgs/%sAUC.png" % y_name)
    plt.show()


# 1. K近邻
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,features_temp,targets,cv=5)
y_name = 'KNN'
print("%s:" % y_name)
print("准确率：",scores.mean())

knn.fit(X_train,y_train)

y_predict = knn.predict(X_test)
# 精准率
print("精准率：",precision_score(y_test,y_predict))
# 召回率
print("召回率：",recall_score(y_test,y_predict))
# F1-Score
print("F1得分：",f1_score(y_test,y_predict))

plotting(knn, y_test, y_name)

# 决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_train,y_train)
y_name = "决策树"

plotting(tree,y_test,y_name)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_name = "随机森林"
plotting(rf,y_test,y_name)

# 逻辑回归
from sklearn.linear_model import LogisticRegression
logic = LogisticRegression(tol=1e-10)
logic.fit(X_train,y_train) 
y_name = "logstic回归"
plotting(logic,y_test,y_name)

# 特征重要性分析
importances = pd.Series(data=rf.feature_importances_,index=features.columns).sort_values(ascending=False)
sns.barplot(y=importances.index,x=importances.values,orient='h')


