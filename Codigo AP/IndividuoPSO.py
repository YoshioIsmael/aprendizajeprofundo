# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:22:03 2019

@author: yoshio
"""
import numpy as np
import random as random

# =============================================================================
# ~ Clase que implementa el indivuo en PSO
# =============================================================================
class IndividuoPSO:
    
    
# =============================================================================
#     ~~Constructor de la clase
# =============================================================================
    def __init__(self, n,acciones):        
        
        #Creamos la posición que es el individuo
        self.posicion = self.generar_individuo_poblacion(n,acciones)
        
        #Iniciamos la velocidad con 1
        self.velocidad = np.array( [1 for i in range(n)] )
        
        #Establecemos la mejor posición hasta el momento
        self.mejorPosicion = self.posicion.copy()
        
        #Establecemos su fitness
        self.fitness   = -10000000
    
    
# =============================================================================
#     ~~Método que permite mover la particula
# =============================================================================
    
    def mover_particula(self, pBest, w, c1, c2):
        v1 = self.multiplicar_vector_por_escalar(w,self.velocidad)
        v2 = self.multiplicar_vector_por_escalar(c1*random.random(), self.restarVectores(self.mejorPosicion,self.posicion))
        v3 = self.multiplicar_vector_por_escalar(c2*random.random(), self.restarVectores(pBest,self.posicion))
        
        self.velocidad = self.sumarVectores(self.sumarVectores(v1,v2),v3)
        self.posicion = self.sumarVectores(self.posicion, self.velocidad)
        
        
            
# =============================================================================
#     ~~Método que permite multiplicar un vector por un escalar
# =============================================================================
    def multiplicar_vector_por_escalar(self,escalar, vector):
        return vector * escalar        
    
    
# =============================================================================
#     ~Método que permite restar 2 vectores
# =============================================================================
    def restarVectores(self, A, B):
        return A - B
    
    
    
# =============================================================================
#     ~Método que permite realizar la suma entre dos vectores
# =============================================================================
    def sumarVectores(self, A, B):
        return A+B
     
# =============================================================================
#         Método que permite crear un vector con numero aleatorios
# =============================================================================
    def _generar_vector_aleatorio(self,tam):        
        return [ random.random() for i in range(tam)]
    
    
# =============================================================================
#     Método que permite crear los sesgos
# =============================================================================
    def _generar_vector_pesos(self,tam):
        return [ random.random() for i in range(tam)]
        
        
# =============================================================================
#    Método que permite generar individuos de la lontitud que deseamos
# =============================================================================
    def generar_individuo_poblacion(self,n,acciones):
       individuo = []  
       individuo.extend(self._generar_vector_aleatorio(6144))
       individuo.extend(self._generar_vector_pesos(32))
       
       individuo.extend(self._generar_vector_aleatorio(32768))
       individuo.extend(self._generar_vector_pesos(64))
       
       individuo.extend(self._generar_vector_aleatorio(36864))
       individuo.extend(self._generar_vector_pesos(64))
       
       individuo.extend(self._generar_vector_aleatorio(2097152))
       individuo.extend(self._generar_vector_pesos(512))
        
       individuo.extend(self._generar_vector_aleatorio(acciones*512))
       individuo.extend(self._generar_vector_pesos(acciones))
       return np.array(individuo)
        