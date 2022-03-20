import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data_dir = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/half-dla_34_mot17_half_dets.txt'
data = np.loadtxt(data_dir)
X = data[:, 2:]
Y = data[:, 0]
mm = MinMaxScaler()
SS = StandardScaler()
X = SS.fit_transform(X)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)


clf = DecisionTreeRegressor()
clf = clf.fit(Xtrain,Ytrain)
score_train = clf.score(Xtrain, Ytrain)
score_test = clf.score(Xtest, Ytest)
y_pred = clf.predict(Xtest)
# MSE_test = mean_squared_log_error(Ytest, y_pred)
MAE_test = mean_absolute_error(Ytest, y_pred)
print('CART training score: ', score_train)
print('CART testing score: ', score_test)
# print('CART MSE: ', MSE_test)
print('CART MAE: ', MAE_test)


plt.plot(y_pred[0:200], label = 'pred', color ='g')
plt.plot(Ytest[0:200], label = 'gt', color = 'r')
plt.legend(loc = 0, ncol = 2)
plt.savefig('result_CART.jpg')

clf = RandomForestRegressor()
clf = clf.fit(Xtrain,Ytrain)
score_train = clf.score(Xtrain, Ytrain)
score_test = clf.score(Xtest, Ytest)
y_pred = clf.predict(Xtest)
# MSE_test = mean_squared_log_error(Ytest, y_pred)
MAE_test = mean_absolute_error(Ytest, y_pred)
print('RF training score: ', score_train)
print('RF testing score: ', score_test)
# print('RF MSE: ', MSE_test)
print('RF MAE: ', MAE_test)

plt.plot(y_pred[0:200], label = 'pred', color = 'b')
plt.legend(loc = 0, ncol = 2)
plt.savefig('result_RF.jpg')