#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:02:54 2019

@author: yoshio
"""

import tensorflow as tf
from ServiciosPoblacion import *

# =============================================================================
#     Método que permite obtener la fotografia del estado
# =============================================================================
def obtener_siguiente_estado(env,action):
    frame_capturado, recompensa, meta, info = env.step(action)
    frame_capturado = frame_capturado[tf.newaxis, ...]                      #agregamos el nuevo eje                    
    frame_capturado = tf.reshape(frame_capturado, shape = [-1, 210,160,3])  # establecemos la forma correcta
    frame_capturado = tf.image.resize(frame_capturado,(64,64))              #reducimos el tamaño
    frame_capturado = tf.cast(frame_capturado, tf.float32)/255.0   
    
    return frame_capturado, recompensa, meta, env


# =============================================================================
#     Método que permite evaluar un individuo, evaluar el vector de pesos.
#     Se establece a la red neuronal y se evalua por completo
# =============================================================================
def evaluar_individuo(pesos_actuales, frames, env,red, evaluaciones = 1):        
    
    pesos_esta   = False   
    puntos_total = 0
    
    for j in range(evaluaciones):
        
        iteraciones  = 0
        action       = 0        
        meta         = False
        
        env.reset()
        
        while(iteraciones < frames):
            
            #Capturamos la foto
            frame_capturado, recompensa, meta, env = obtener_siguiente_estado(env,action)
            
            
            #se lo enviamos a la red y nos regresa el resultado
            resultado = red(frame_capturado)   
            
            #arreglar esto!!!
            if not pesos_esta:
                #se establecen a la red                
                red.establecer_pesos(convertir_vector_a_parametros(pesos_actuales,env.action_space.n))                  
                pesos_esta = True                 
            
            #elegimos la opción a realizar
            action = tf.keras.backend.eval(tf.math.argmax(resultado[0]))                
            
            #incrementamos las iteraciones en uno
            iteraciones  += 1
            puntos_total += recompensa            
            
            #Si ya llegamos a la meta, nos salimos del ciclo
            if meta:
                break
                    
    return (puntos_total, iteraciones,env)