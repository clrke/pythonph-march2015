from sklearn.datasets import load_digits

digits = load_digits()

print len(digits.images)

import matplotlib.pyplot as plt

for i in range(100):
	plt.subplot(10, 10, i+1)
	plt.axis('off')
	plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()

from sklearn.svm import SVC

classifier = SVC().fit(digits.data, digits.target)

random_indices = [123, 456, 789, 1059, 1289, 1567]

random_digits_data = [digits.data[i] for i in random_indices]
random_digits_images = [digits.images[i] for i in random_indices]

predictions = classifier.predict(random_digits_data)

for i in range(6):
	plt.subplot(2, 3, i+1)
	plt.axis('off')
	plt.imshow(random_digits_images[i], cmap=plt.cm.gray_r,
		interpolation='nearest')
	plt.title(predictions[i])

plt.show()
