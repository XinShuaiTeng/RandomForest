import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report

df = pd.read_csv("./data/process_heart.csv")
print(df.shape)
print(df.head())

# 取出目标值的一列
X = df.drop('target', axis=1)
y = df['target']
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# 100颗决策树集成
model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=5)
model.fit(X_train, y_train)

print(X_test.shape)
# 选择一个测试样本
test_sample = X_test.iloc[2]
print(test_sample.shape)
# 转换样本维度
test_sample = np.array(test_sample).reshape(1,-1)
# print(test_sample)

# 对未知样本进行预测
# print(model.predict(test_sample))
# print(model.predict_proba(test_sample))

# 对测试集上所有数据进行预测
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

print(y_pred)
print(y_test)
print(y_pred_prob)

def cnf_martix_plotter(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Confusion Martix')
    plt.colorbar()
    tick_markes = np.arange(len(classes))
    plt.xticks(tick_markes, classes, rotation=0)
    plt.yticks(tick_markes, classes, rotation=45)

    threshold = cm.max() /2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black', fontsize = 25)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predict Label")
    plt.show()

# 混淆矩阵
confusion_matrix_model = confusion_matrix(y_test, y_pred)
cnf_martix_plotter(confusion_matrix_model, ["Healthy","Disease"])

# 评估报告
print(classification_report(y_test, y_pred, target_names=["Healthy","Disease"]))



# 查看决策树具体信息
# estimator = model.estimators_[7]
# print(estimator)

feature_names = X_train.columns
y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values
# print(y_train_str)
# print(feature_names)

# export_graphviz(estimator, out_file='tree.dot', feature_names = feature_names, class_names = y_train_str,
#                 rounded=True, proportion=True,
#                 label='root', precision = 2, filled=True)
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#
# Image(filename='tree.png')

print("特征排序")
feature_names = X_test.columns
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

for index in indices:
    print("features %s (%f)" % (feature_names[index], feature_importances[index]))

plt.figure(figsize=(16,8))
plt.title("Feature Importance")
plt.bar(range(len(feature_importances)),feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='b', rotation = 90)
plt.savefig("./imgs/importance.png")
plt.show()

