import pandas as pd
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/heart.csv')

# 使用第三方包分析数据集，并将报告保存至本地
# profile = pandas_profiling.ProfileReport(df)
# profile.to_file("./profile.html")

# 检测是否有缺失值
print(df.isnull().sum())
# 查看数据类型
print(df.dtypes)

# 检测相关性，颜色越亮越相关
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f',square=True)
plt.savefig("./imgs/corr.png")
plt.show()

sns.displot(df['age'])
# plt.savefig("./imgs/displot.png")
plt.show()


sns.countplot(x='target', data=df, palette='bwr')
plt.savefig("./imgs/countplot-target.png")
plt.show()

sns.countplot(x='sex', data=df, palette='mako_r')
plt.xlabel("Sex(0=female,1=male)")
plt.savefig("./imgs/countplot-sex.png")
plt.show()

# 不同年龄段与患病相关关系
pd.crosstab(df.age, df.target).plot(kind='bar', figsize=(20,6))
plt.title("Heart Disease Frequency for Ages")
plt.xlabel("Age")
plt.ylabel("Frequency")
# plt.savefig("./imgs/age-frequency.png")
plt.show()

# 箱型图
sns.boxplot(x=df['target'], y=df['age'])
plt.show()

# 小提琴图
sns.violinplot(x=df['target'], y=df['age'])
# plt.savefig("./imgs/violin.png")
plt.show()

# 不同年龄段与性别相关关系
pd.crosstab(df.sex, df.target).plot(kind='bar', figsize=(15,6))
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("Sex(0=female,1=male)")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel("Frequency")
plt.show()

# 不同最大心率，不同年龄，患心脏病和不患心脏病的分布
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)],c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)],c="blue")
plt.title("Heart Disease for Age and Maximum Heart Rate")
plt.legend(["Diease","No Diease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
# plt.savefig("./imgs/rate-age.png")
plt.show()

print(df.head())

# 数据离散化
# 定类的特征 和 定序特征应该转化为object类型
df.loc[df['sex'] == 0,'sex'] = 'female'
df.loc[df['sex'] == 1,'sex'] = 'male'

df.loc[df['cp'] == 0, 'cp'] = 'typical angina'
df.loc[df['cp'] == 1, 'cp'] = 'atypical angina'
df.loc[df['cp'] == 2, 'cp'] = 'non-anginal angina'
df.loc[df['cp'] == 3, 'cp'] = 'asymptomatic'

df.loc[df['fbs'] == 0, 'fbs'] = 'lower than 120mg/ml'
df.loc[df['fbs'] == 1, 'fbs'] = 'higher than 120mg/ml'

df.loc[df['restecg'] == 0, 'restecg'] = 'normal'
df.loc[df['restecg'] == 1, 'restecg'] = 'ST-T wave abnormality'
df.loc[df['restecg'] == 2, 'restecg'] = 'left ventricular hypertrophy'

df.loc[df['exang'] == 0, 'exang'] = 'no'
df.loc[df['exang'] == 1, 'exang'] = 'yes'

df.loc[df['slope'] == 0, 'slope'] = 'upsloping'
df.loc[df['slope'] == 1, 'slope'] = 'flat'
df.loc[df['slope'] == 2, 'slope'] = 'downsloping'

df.loc[df['thal'] == 0, 'thal'] = 'unknown'
df.loc[df['thal'] == 1, 'thal'] = 'normal'
df.loc[df['thal'] == 2, 'thal'] = 'fixed defect'
df.loc[df['thal'] == 3, 'thal'] = 'reversable defect'

print(df.head())
print(df.dtypes)

# one-hot 编码
df = pd.get_dummies(df)
print(df.columns)
print(df.head())

df.to_csv("./data/process_heart.csv", index=False)
