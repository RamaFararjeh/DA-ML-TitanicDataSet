import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,confusion_matrix


# Read Data 
df=pd.read_csv(r'C:\Users\admin\Development\Classification\Machine-Learning\Data-Analysis\Titanic-DataSet\titanic_train.csv')

# print(df.head())
# print(df.info())
# print(df.describe())


# Drop column's Features.
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True) # cuz have multi of null value

# drop row Features.
d_age=df[df['Age']<=1].index.to_list()
df.drop(index=d_age,inplace=True) # drop for rows


# print(df.head())
# print(df.isna().sum())
# Age => 177 missing Value
# Embarked => 2 missing Value

# ======================================================================

# Separating categorical and numarical columns
cat_data=df.select_dtypes(include=['object']).columns
num_data=df.select_dtypes(include=['int64', 'float64']).columns
print(f'Categorical Data : \n {cat_data}') # as list
print(f'numerical data : \n {num_data}') # as list

# ======================================================================

# Check for unique Value for categorical Data. 
def unique_data():
    print('*'*100)
    print('Let check for Unique data.')
    # my_list=categorical_data
    my_list=cat_data
    print(f'Name of Col. {my_list}')
    len_col=len(my_list)
    for i in my_list:
        print(f'number of unique [ {i} ] : {df[i].nunique()} ')
        print('Unique Value:')
        print(f'Value of unique [ {i} ] : {df[i].unique()} ')
        print()
        print(df[i].value_counts())  
        print('-----------------------------------------------------------------------')


# unique_data()

# ======================================================================

# Check for Number of Null [ Missing Value ]
def isnull_data():
    print('*'*100)
    print('Let check for missing data.')
    print(f'Number of Missing Data =\n{df.isna().sum()}')

# isnull_data()

# ======================================================================

# Histogram for Distribution.

def hist_plot():
    for i in num_data:
        sns.histplot(df[i],bins=20,kde=True)
        plt.hist(df[i],bins=20,edgecolor='k')
        plt.xlabel(i)
        plt.ylabel('Frequncy')
        plt.title(f'Histogram of  {i}')
        plt.show()


# ======================================================================

# Box plot for Outliers.

def box_plot():
    # plt.boxplot(df[num_col]) # this true.
    sns.boxplot(df[num_data]) # and this true but it better.
    plt.xlabel(num_data.to_list())
    plt.ylabel('Frequancy')
    plt.title(f'Box plot of {num_data.to_list()}')
    plt.show()

# ======================================================================

# Heatmap:-
# to make heatmap we must drop all categorical data , so befor running the command we should activate cat_data().
#  or use heatmap for numarical data only .

# Calculate correlation matrix
correlation_matrix = df[num_data].corr()
print(f'correlation_matrix : {correlation_matrix}')
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix,annot=True,linecolor='red',cmap='Greens',fmt=".2f",center=0) # color-heatmap : BuPu , Greens , YlGnBu , Blues
plt.show()


# categorical_columns
def cat_pie():
    for i in cat_data:
        countt=df[i].value_counts()
        print(countt)
        plt.figure(figsize=(8,6))
        plt.pie(countt,labels=countt.index,autopct="%1.1f%%")#,autopct="%1.1f%%",startangle=140
        plt.title(f"pie plot {i}")
        plt.show()

# Scatter Plot

survived_df = df[df['Survived'] == 1]
not_survived_df = df[df['Survived'] == 0]
print('Split each class in the target features :')
print(survived_df)
print(not_survived_df)


# EDA CUZ have Classification Data.
# scatterr_survived_df
def scatterr_survived_df():
    # split num col only 
    num_features=survived_df.select_dtypes(include=['number']).columns # as a list 
    for i in num_features:
        plt.figure(figsize=(8,6))
        plt.scatter(survived_df[i],survived_df['Survived'],c='red',marker='o',label='Data Points')
        plt.scatter(not_survived_df[i], not_survived_df['Survived'], c='blue', marker='o', label='Survived=0', alpha=0.7)

        # Marker Value :
        # 'o' for circular.
        # 's' for square.
        # '^' for triangle.

        # label  => legend

        plt.xlabel(i)
        plt.ylabel('Survived')
        plt.title(f'Scatter Plot {i} vs Survived 0 and 1 class.')

        plt.legend() # مفتاح الخريطه
        plt.show()

# ====================================================================================

# Interactive plot
ct=pd.crosstab(df['Pclass'],df['Survived'])
ct.reset_index(inplace=True)
melt_data=pd.melt(ct,id_vars='Pclass',value_vars=[0,1],
                    var_name='Survived',value_name='Count')

# creat interactive bar plot
fig=px.bar(melt_data,x='Pclass',y='Count',
            color='Survived',title='Survived by Passenger class',
            labels={'Pclass': 'Passenger class', 'Count':' Number of Passenger '},
                    text='Count')

# fig.show()


# Interactive plot
ct=pd.crosstab(df['SibSp'],df['Survived'])
ct.reset_index(inplace=True)
melt_data=pd.melt(ct,id_vars='SibSp',value_vars=[0,1],
                    var_name='Survived',value_name='Count')

# creat interactive bar plot
fig=px.bar(melt_data,x='SibSp',y='Count',
            color='Survived',title='Survived by Sibbling Number',
            labels={'SibSp': 'SibSp class', 'Count':' Number of Sibbling '},
                    text='Count')

# fig.show()


# Interactive plot
ct=pd.crosstab(df['Parch'],df['Survived'])
ct.reset_index(inplace=True)
melt_data=pd.melt(ct,id_vars='Parch',value_vars=[0,1],
                    var_name='Survived',value_name='Count')

# creat interactive bar plot
fig=px.bar(melt_data,x='Parch',y='Count',
            color='Survived',title='Survived by Parent child class',
            labels={'Parch': 'Parent child class', 'Count':' Number of Parnet & nchild '},
                    text='Count')

# fig.show()


# Convert all categorical col to numerical col using Label Encoder.

label_encoder = LabelEncoder() # instance of LabelEncoder class
columns_to_encode = ['Sex','Embarked']
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

print(f'Encoded Data : \n {df}')

# ======================================================================

# hist_plot()
# box_plot()
# cat_pie()
scatterr_survived_df()

# ======================================================================

# split train data  before fill any null value [ "split before filling" approach is often recommended for model evaluation  ]
x=df.drop('Survived',axis=1) # x => new DataFrame without salary features.
y=df['Survived']  # y => have the predict target variable.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) # had mo kader afhamo :(


# ======================================================================

# Create a SimpleImputer for filling missing values with the median
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the 'Age' column of the training data
imputer.fit(x_train['Age'].values.reshape(-1, 1))

# Transform the 'Age' column of both the training and test data using the fitted imputer
x_train['Age'] = imputer.transform(x_train['Age'].values.reshape(-1, 1))
x_test['Age'] = imputer.transform(x_test['Age'].values.reshape(-1, 1))

# print(x_train)
# print(x_train.isna().sum())

# ======================================================================

# Scaling 

scaler=MinMaxScaler() # instance of MinMaxScaler class.
feature_columns_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
# Fit
scaler.fit(x_train[feature_columns_num])
# Transform
# x_train_scaled=x_train.copy()
# x_test_scaled=x_test.copy()
x_train[feature_columns_num]=scaler.transform(x_train[feature_columns_num])
x_test[feature_columns_num]=scaler.transform(x_test[feature_columns_num])
print(x_train.head())
# print(x_train.isna().sum())

# ======================================================================

# Logistic Regression Model 

# Creat Model
lr_model=LogisticRegression()

# Train Model 
lr_model.fit(x_train,y_train)

# Evaluate the Model
lr_score=lr_model.score(x_test,y_test)

print(f'Logistic Regression accuracy : {lr_score}')
# ======================================================================


# KNN Model

# Creat Model
knn_model=KNeighborsClassifier()

# Train the Model
knn_model.fit(x_train,y_train)

# Evaluate the Model
knn_Score=knn_model.score(x_test,y_test)
print(f'KNN accuracy : {knn_Score}')



# ======================================================================

# SVC Model

# Creat Model 
svc_model=SVC()

# Train the Model
svc_model.fit(x_train,y_train)

#Evaluate the Model
svc_score=svc_model.score(x_test,y_test)

# print(f'SVC accuracy : {svc_score}')

# Generate a classification report
y_predict=svc_model.predict(x_test)
report_svc=classification_report(y_test,y_predict)
print(f'SVC Report \n {report_svc}')

# ======================================================================


# Accuracy 
accuracy=accuracy_score(y_test,y_predict)
print(f'Accuracy : {accuracy}')

# Precision
precision=precision_score(y_test,y_predict,average='weighted')
print(f'Precision : {precision}')

# Recall
recall=recall_score(y_test,y_predict,average='weighted')
print(f'Recall : {recall}')

# F1 Score
f1=f1_score(y_test,y_predict,average='weighted')
print(f'F1 Score : {f1}')

# Confussion Matrix
conf_matrix=confusion_matrix(y_test,y_predict)
print(f'Confussion Matrix : {conf_matrix}')




# ======================================================================


# insight:
# 1) target col. => Survived
# 2) Age 714 / 891 missing value => 177       Done with fill using median
# 3) Cabin 204 / 891 missing value => 687     Done with drop
# 4) Embarked 889 / 891 missing value => 2    Done with mode
# 5) There is no Dublicated Value
# 6) Features you can drop => PassengerId / Name / Ticket / 
# 7) in pclass the third class is more than other i mean more than 1 class and 2 class.
# 8) Most of the people participating are between approximately 20 and 40 years old.
# 9) The percentage of people who died is more than the percentage of people who are alive.
# 10) Survived    Pclass      SibSp     Parch      => this is class's
# 11) Fare Features skwed to the Right.

# Categorical Data : Sex  |  Embarked


# ======================================================================
# Rama Fararjeh