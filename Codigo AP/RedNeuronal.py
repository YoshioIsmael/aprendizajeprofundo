# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:32:17 2019

@author: yoshio
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Model


# =============================================================================
# Clase que implementa una red neuronal
# =============================================================================
class RedNeuronal(Model):
    
    
# =============================================================================
#     ~Constructor de la clase
# =============================================================================
    def __init__(self, salidas_posibles = 18):
        
        #llamamos al constructor de la clase padre
        super(RedNeuronal, self).__init__()
        
        #definimos nuestras capas         
        self.conv1 = Conv2D(filters = 32, kernel_size = 8, strides = 4, activation = tf.nn.relu, padding = "same")
        self.conv2 = Conv2D(filters = 64, kernel_size = 4, strides = 2, activation = tf.nn.relu, padding = "same")
        self.conv3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = tf.nn.relu, padding = "same")
        self.flat  = Flatten() 
        self.fc    = Dense(512)
        self.out   = Dense(salidas_posibles)
        
        
# =============================================================================
#     Método que permite realizar la evaluación de la red    
# =============================================================================
    def call(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flat(x)
        
        x = self.fc(x)
        x = tf.nn.relu(x)
        x = self.out(x)        
        x = tf.nn.softmax(x)
        
        return x
        
    
# =============================================================================
#     ~~Método que permite obtener los pesos de la primera capa
# =============================================================================
    def obtenerPesos(self):        
        #creamos un diccionario de los datos 
        pesos = {'cv1' : self.conv1.get_weights(),
                 'cv2' : self.conv2.get_weights(),
                 'cv3' : self.conv3.get_weights(),
                 'fc1' : self.fc.get_weights(),
                 'out' : self.out.get_weights()}        
        return pesos
    
    
# =============================================================================
#     ~Método que permite establecer los pesos a la red neuronal
# =============================================================================
    def establecer_pesos(self, red): 
        self.conv1.set_weights(red['cv1'])
        self.conv2.set_weights(red['cv2'])
        self.conv3.set_weights(red['cv3'])
        self.fc.set_weights(red['fc1'])
        self.out.set_weights(red['out'])
        