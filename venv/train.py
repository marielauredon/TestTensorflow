import sys
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam

from model import simpleCNN

#model.summaury() ->>pour le résumé du modèle
model=simpleCNN.simplecnn()

#compilation du modèle
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

#On importe la dataset (base de donnée)
from keras.datasets import fashion_mnist
((images, targets),(testX,testY))=fashion_mnist.load_data()
#images=images[:10000]
#targets=targets[:10000]
print(images.shape)
print(targets.shape)

#On crée la liste des catégories
targets_names=["T-shirt/top","Trouser/pants","Pullover shirt","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

#Afficher une image
plt.imshow(images[0], cmap="binary")
plt.title(targets_names[targets[0]])
plt.show()

if K.image_data_format()=="channels_first":
    images = images.reshape((images.shape[0],1,28,28))
    testX = testX.reshape((testX.shape[0],1,28,28))
else:
    images = images.reshape((images.shape[0],28,28,1))
    testX = testX.reshape((testX.shape[0],28,28,1))

#Normalisation (pour la facilitation du calcul des paramètres du réseau de neurones)
images = images.astype("float32")/255.0
testX=testX.astype("float32")/255.0

#conversion sous forme de vecteur
targets = np_utils.to_categorical(targets,10)
testY=np_utils.to_categorical(testY,10)

#Apprentissage
#batch_size = taille de lot et epochs = nb d'itération
Hist=model.fit(images,targets,validation_data=(testX,testY),batch_size=32, epochs=15)
print(Hist)


model.save('model.h5')
#prediction
#model_output=model.predict(images[0:1])
#print(model_output, targets[0:1])

