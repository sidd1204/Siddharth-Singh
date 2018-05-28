import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
data = pd.DataFrame(boston.data, columns = boston.feature_names)
data.head()
data.shape
boston.target
boston.DESCR
data["Target"]=boston.target
data.head(20)
data.shape
data.isnull().sum()
data.describe()

pd.crosstab(pd.cut(data.RM,bins= 5), pd.cut(data.Target,bins=5))


data.corr()
data.ZN.value_counts()
pd.crosstab(data.CHAS, pd.cut(data.Target,bins=5))
pd.crosstab(pd.cut(data.ZN,bins=5), pd.cut(data.Target,bins=5))
plt.hist(data.ZN,bins= 5)

data.Target.mean()

X= data.iloc[:,:-1]
Y= data.iloc[:,-1]



from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

Y_test_pred=LR.predict(X_test)
LR.coef_

from sklearn import metrics
MSE = metrics.mean_squared_error(Y_test,Y_test_pred)
MSE
R2= metrics.r2_score(Y_test,Y_test_pred)
R2
LR.score(X_test,Y_test)
