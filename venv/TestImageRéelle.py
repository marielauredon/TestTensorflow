import numpy as np
from matplotlib import pyplot as plt

import cv2
from keras.models import load_model

model=load_model('model.h5')

#Chargement de l'image, filtrage et conversion d'espace colorimétrique
img=cv2.imread('robe bleu.jpg')
imgfilt=cv2.GaussianBlur(img,(3,3),0)
imgray=cv2.cvtColor(imgfilt,cv2.COLOR_BGR2HSV)

#Définition de la couleur de sélection (celle de l'objet)
low=np.array([115,100,20])
high=np.array([125,255,255])
masq=cv2.inRange(imgray,low,high)

#Sélection du vêtement à partir de la couleur
sel=cv2.bitwise_and(imgray,imgray,mask=masq)

#Redimensionnement de l'image
redim=cv2.resize(cv2.cvtColor(sel, cv2.COLOR_BGR2GRAY),(28,28), interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(5,5))
plt.imshow(imgray)

plt.figure(figsize=(5,5))
plt.imshow(sel,cmap='gray')

#redimensionnement de l'images
out=redim.reshape((28,28,1))
out=out.astype("float32")/255.0

#out=(out*255).astype("uint8")
#mettons l'image dans un tableau numpy
test=[]
test.append(out)
out=np.asarray(test)

#définiton des labels
targets_names=["T_shirt", "pantalon", "pull-over", "robe", "manteau", "sandale", "chemise", "basket", "sac", "bottine"]

result=model.predict(out)
print(result)
prediction=result.argmax(axis=1)
nom=targets_names[prediction[0]]
percent = "{:0.2f}%".format(np.float(100*result[0][prediction[0]]))

#Afficher image et résultat
couleurTXT=(0,255,0)#si prédiction bonne
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.putText(img,nom,(5,5),cv2.FONT_HERSHEY_SIMPLEX,0.25,couleurTXT,1)

#Ajout de la probabilité
cv2.putText(img,percent,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.25,couleurTXT,1)
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

plt.show()