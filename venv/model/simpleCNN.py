import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import keras

#Création du 1er modèle
#mise en place d'un réseau de neurones classiques
#Transformer l'image en 1 vecteur
def simplecnn():#ccn=Convolutional Neural Network ou réseau de neurones convolutifs
    model=keras.models.Sequential() #modède de type séquence
#ajout couches convolutives
    model.add(keras.layers.convolutional.Conv2D(filters=32, kernel_size = (5,5), padding="same",input_shape=(28,28,1)));
#couche d'activation
    model.add(keras.layers.core.Activation("relu"))
#couches de convolution supplémentaires
    model.add(keras.layers.convolutional.Conv2D(filters=32,kernel_size = (5,5), padding="same", activation="relu"))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#dropout pour utiliser 80% des neuronnes pour éviter le surapprentissage
    model.add(keras.layers.core.Dropout(0.2))
#ajout de couches avec plus de filtres (détectent des caractéristiques plus complexes)
    model.add(keras.layers.convolutional.Conv2D(filters=64,kernel_size = (5,5), padding="same", activation="relu"))
    model.add(keras.layers.convolutional.Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.core.Dropout(0.2))
#on ajoute une opération au modèle
    model.add(keras.layers.core.Flatten())
#Ajout des couches de neurones
    model.add(keras.layers.core.Dense(256,activation='relu'))
    model.add(keras.layers.core.Dense(128,activation='relu'))
    model.add(keras.layers.core.Dense(10,activation="softmax"))
    return model
