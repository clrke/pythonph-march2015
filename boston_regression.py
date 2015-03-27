from sklearn.datasets import load_boston
from sklearn.svm import SVR
import matplotlib.pyplot as plt

boston = load_boston()

regressor = SVR(gamma=0.001).fit(boston.data, boston.target)
prediction = regressor.predict(boston.data)

plt.plot(prediction, 'r')
plt.plot(boston.target, 'b')

plt.show()

print regressor.score(boston.data, boston.target)
