from your_code import get_animal_features as gaf

animals = ['animal1.jpg', 'animal2.jpg', 'animal3.jpg',
			'animal4.jpg', 'animal5.jpg', 'animal6.jpg']

features = list(gaf(animals))
print features

classes = [1, 0, 1, 0, 1, 0]
print classes

from matplotlib import pyplot as plt

x = [feature[0] for feature in features]
y = [feature[1] for feature in features]

for plot in zip(x, y, classes):
	color = 'bo' if plot[2] == 1 else 'ro'
	plt.plot(plot[0], plot[1], color)

plt.axis([-0.2, 1.2, -0.2, 1.2])
plt.show()

from sklearn.svm import SVC
classifier = SVC().fit(features, classes)

animal1 = [0, 0.6]
animal2 = [1, 0.3]
animal3 = [1, 0.5]
animal4 = [0, 0.3]

prediction = classifier.predict([animal1, animal2, animal3, animal4])

print prediction
