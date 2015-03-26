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

classifier = SVC(gamma=0.001).fit(digits.data, digits.target)

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

# scoring
random_digits_target = [digits.target[i] for i in random_indices]
print classifier.score(random_digits_data, random_digits_target)

# real scoring
training_data = digits.data[:1000]
training_target = digits.target[:1000]

test_data = digits.data[1000:]
test_target = digits.target[1000:]

classifier = SVC(gamma=0.001).fit(training_data, training_target)
print classifier.score(test_data, test_target)
