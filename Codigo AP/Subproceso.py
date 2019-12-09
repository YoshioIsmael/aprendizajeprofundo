#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:09:53 2019

@author: yoshio
"""

import threading
import gym 
import RedNeuronal
import tensorflow as tf

from Evaluador import *

# =============================================================================
# Clase que nos permite paralelizar los individos
# =============================================================================
class EvaluacionParalela(threading.Thread):
    
# =============================================================================
#     Constructor de la clase
# =============================================================================
    def __init__(self,ambiente, acciones_posibles, corridas, individuo, frames, Name=None,pso = True):
        super().__init__(group=None, target=None, name=Name,daemon=True)
        self.invididuo = individuo
        self.puntos    = 0
        self.red       = RedNeuronal.RedNeuronal(acciones_posibles)               
        self.env       = gym.make(ambiente)
        self.corridas  = corridas 
        self.frames    = frames
        self.pso       = pso
        self.iteraciones = 0
        self.name      = Name
        
    
        
        
# =============================================================================
#         Método que nos permite ejecutar el contenido del hilo
# =============================================================================
    def run(self):        
        individuo = []
        if self.pso:
            self.invididuo, _ = self.invididuo
            individuo         = self.invididuo.posicion
            
        else:
            self.invididuo,_ = self.invididuo
            individuo        = self.invididuo
            
        for i in range(self.corridas):
            (evaluacion, iteraciones , env) = self.evaluar_individuo(individuo,self.frames, self.env,self.red)
            self.puntos += evaluacion
            self.iteraciones += iteraciones
            
        self.env.close()
    
# =============================================================================
#     Método que permite obtener los resultados de la red
# =============================================================================
    def resultados(self):
        return (self.invididuo,(self.puntos/self.corridas),self.iteraciones )
    
    
    # =============================================================================
    #     Método que permite obtener la fotografia del estado
    # =============================================================================
    def obtener_siguiente_estado(self,env,action):
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
    def evaluar_individuo(self,pesos_actuales, frames, env,red):        
        action       = 0
        pesos_esta   = False
        meta         = False
        iteraciones  = 0
        puntos_total = 0
        env.reset()
        
        while(iteraciones < frames):
            
            #Capturamos la foto
            frame_capturado, recompensa, meta, env = self.obtener_siguiente_estado(env,action)
            
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