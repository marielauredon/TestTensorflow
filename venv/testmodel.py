import numpy as np
from keras.datasets import fashion_mnist
from keras import backend as K
from keras.utils import np_utils
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model

model=load_model('model.h5')

#Définition des tables
targets_names=["T_shirt", "pantalon", "pull-over", "robe", "manteau", "sandale", "chemise", "basket", "sac", "bottine"]

#chargement du jeu de données
((images, targets),(testX,testY))=fashion_mnist.load_data()
#vérification du format de TensorFlow
if K.image_data_format()=="channels_first":
    testX = testX.reshape((testX.shape[0],1,28,28))
else:
    testX = testX.reshape((testX.shape[0],28,28,1))

#Conditionnement des données
testX=testX.astype("float32")/255.0
testY=np_utils.to_categorical(testY,10)

#création de la liste des images tests à afficher
tabimages=[]

#Extraction aléatoire et test du modèle
for i in np.random.choice(np.arange(0,len(testY)), size=(16,)):
    results=model.predict(testX[np.newaxis,i])
    prediction=results.argmax(axis=1)
    label=targets_names[prediction[0]]

#Extraction de l'image test X en niveaus de gris suivant la configuration TensorFlow
    if K.image_data_format()=="channels_first":
        image=(testX[i][0]*255).astype("uint8")
    else:
        image = (testX[i]*255).astype("uint8")

    couleurTXT=(0,255,0)#si prédiction bonne

    if prediction[0]!=np.argmax(testY[i]):
        couleurTXT=(255,0,0)#si prédiction mauvaise

#fusion des 3 canaux d'images et superposition du label
    image=cv2.merge([image]*3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image,label,(5,20), cv2.FONT_HERSHEY_SIMPLEX,0.75,couleurTXT,2)
    tabimages.append(image)

#Affichage des images tests
plt.figure(figsize=(7,7))
for i in range (0,len(tabimages)):
    plt.subplot(4,4,i+1)
    plt.imshow(tabimages[i])

plt.show()