import pandas as pd
col_names=['time','task','involvement','friendly','classtime','environment','tired','distract','concentrate','disconfort','sleep','argue','emotion','specs','device','class','work','pain','internet','family','attack','attach','SCORE']
data=pd.read_csv("C:/Users/User/OneDrive/Desktop/data_col_con.csv")
print(data)
data.head()

feature_cols = ['time','task','involvement','friendly','classtime','environment','tired','distract','concentrate','discomfort','sleep','argue','emotion','specs','device','class','work','pain','internet','family','attacks','attach',' SCORE']
X = data[feature_cols] # Features
y = data.LABEL #Target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)



#
y_pred=logreg.predict(X_test)

import pickle
pickle.dump(logreg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
#ROC
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#single class prediction





