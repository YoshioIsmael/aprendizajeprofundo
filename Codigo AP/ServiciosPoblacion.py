"""
Created on Thu Oct 24 09:29:25 2019

@author: yoshio
"""

import random
import numpy as np
import IndividuoPSO as pso
from archivos import *


# =============================================================================
# Métodos que permiten trabajar con la población, generarla y modificar su estructura
# =============================================================================



# =============================================================================
# Este método nos permite desenvolver los pesos y los metemos en un vector de pesos
# =============================================================================
def obtener_pesos_convolucional(lista, pesos):       
   for i in lista:
       for j in i:
           for k in j:           
               for l in k:  
                   pesos.append(l)
   return pesos


# =============================================================================
# Método que permite obtener el vector de pesos de una red convolucional  
# =============================================================================
def obtener_convolucional_pesos(pesos,configuracion):
   a,b,c = configuracion 
   indice = 0              
   res = []
   for i in range(a):           
       v1 = []
       for j in range(a):               
           v2 = []
           for k in range(b):                   
               ve = []
               for l in range(c):                       
                   ve.append(pesos[indice])
                   indice += 1
               v2.append(ve)
           v1.append(v2)
       res.append(v1)       
   return np.array(res)
   

# =============================================================================
# Método que permite obtener los pesos de una capa de neuronas
# =============================================================================
def obtener_pesos_vector(pesos):       
   w = []       
   for  i in pesos:
       w.append(i) 
   return np.array(w)
   
    
# =============================================================================
# Método que permite obtener los sesgos de una capa
# =============================================================================
def obtener_pesos_bias(lista, pesos):       
   for i in lista:
       pesos.append(i)
   return pesos


# =============================================================================
# Método que noss permite obtener los pesos completos de una capa convolucional
# =============================================================================
def de_convolucional_a_vector(capa_convolucional, pesos):
  #obtenemos la primera posición de nuestro arreglo
  pesos = obtener_pesos_convolucional(capa_convolucional[0], pesos)
  #Obtenemos los sesgosde la capa
  pesos = obtener_pesos_bias(capa_convolucional[1], pesos)
  return pesos


# =============================================================================
# Método que permite obtener los pesos de una capa completamente conectada
# =============================================================================
def obtener_pesos_densa(lista, pesos):        
    for i in lista:
        for j in i:        
            pesos.append(j)     
    return pesos
 
    
# =============================================================================
# Método que permite obtener los pesos de una capa densa, a partir de un vector de pesos
# =============================================================================
def obtener_pesos_vector_densa(pesos,configuracion):
   a, b   = configuracion
   indice = 0
   ve     = []
   
   for i in range(a):
       v1 = []
       for j in range(b):
           v1.append(pesos[indice])
           indice += 1
       ve.append(v1.copy())
   
   return np.array(ve)

# =============================================================================
# Método que nos permite obtenemos los pesos de una capa densa
# =============================================================================
def de_densa_a_vector(capa_densa,pesos):    	
	#Obtenemos los pesos de la capa densa
	pesos = obtener_pesos_densa(capa_densa[0], pesos)
    #obtenemos los sesgos de la capa
	pesos = obtener_pesos_bias(capa_densa[1], pesos)

	return pesos
    
# =============================================================================
# Método que permite obtener los pesos de una red completa
# =============================================================================
def obtener_pesos_red(red):
   pesos = []
   
   #obtenemos una capa de la red
   print('imprimimos caracteristicas',red['out'][0].shape)
   return pesos
   
    
# =============================================================================
#     Método que permite comvertir un vector de pesos a la estructura especial para la 
#     red neuronal
# =============================================================================
def convertir_vector_a_parametros(vector_pesos,acciones):
    cv1 = []
    cv1.append(obtener_convolucional_pesos(vector_pesos[0:6144],(8,3,32)))
    cv1.append(obtener_pesos_vector(vector_pesos[6144:6176]))
    
    cv2 = []
    cv2.append(obtener_convolucional_pesos(vector_pesos[6176:38944],(4,32,64)))
    cv2.append(obtener_pesos_vector(vector_pesos[38944:39008]))
    
    cv3 = []
    cv3.append(obtener_convolucional_pesos(vector_pesos[39008:75872],(3,64,64)))
    cv3.append(obtener_pesos_vector(vector_pesos[75872:75936]))
    
    fc = []
    
    fc.append(obtener_pesos_vector_densa(vector_pesos[75936:2173088],(4096,512)))
    fc.append(obtener_pesos_vector(vector_pesos[2173088:2173600]))        
    
    out = []
    
    _acciones = acciones*512
    out.append(obtener_pesos_vector_densa(vector_pesos[2173600:(2173600 + _acciones)],(512,acciones)))
    out.append(obtener_pesos_vector(vector_pesos[_acciones:(_acciones + acciones)]))
    
    parametros = {'cv1':cv1, 'cv2':cv2, 'cv3':cv3, 'fc1':fc, 'out':out }
    
    return parametros    
           
# =============================================================================
#         Método que permite crear un vector con numero aleatorios
# =============================================================================
def _generar_vector_aleatorio(tam):        
    return [ random.random() for i in range(tam)]

    
# =============================================================================
#     Método que permite crear los sesgos
# =============================================================================
def _generar_vector_pesos(tam):
    return [ random.random() for i in range(tam)]
    
# =============================================================================
# Permite henerar la poblacion inicial de individuso
# =============================================================================
def generar_poblacion_genetico(poblacion,acciones, cargar = False,nombre="elite.p"):
    poblacion_inicial = []     
    for i in range(poblacion):
        
        if random.random() > .5 and cargar:
            poblacion_inicial.append(cargar_ultimo_elite(nombre))    
        else:            
            poblacion_inicial.append((generar_individuo(acciones),0))
        
    return poblacion_inicial


# =============================================================================
# Genera la poblacion inicial de individuos
# =============================================================================
def generar_individuo(acciones):
    individuo = []
    individuo.extend(_generar_vector_aleatorio(6144))
    individuo.extend(_generar_vector_pesos(32))
   
    individuo.extend(_generar_vector_aleatorio(32768))
    individuo.extend(_generar_vector_pesos(64))
   
    individuo.extend(_generar_vector_aleatorio(36864))
    individuo.extend(_generar_vector_pesos(64))
   
    individuo.extend(_generar_vector_aleatorio(2097152))
    individuo.extend(_generar_vector_pesos(512))
    
    individuo.extend(_generar_vector_aleatorio(acciones*512))
    individuo.extend(_generar_vector_pesos(acciones))
    
    return np.array(individuo)

# =============================================================================
# Método que permite mutar a un individuo
# =============================================================================
def mutar_individuo(arreglo,mult):
    ruido = random.random()
    _arreglo,b = arreglo    
    for i in range(len(_arreglo)):
        _arreglo[i] = _arreglo[i] + mult * ruido
        
    return (_arreglo, b)
    
# =============================================================================
#  ~Método que permite generar la poblacion de pso
# =============================================================================
def generar_poblacion_pso(numero_poblacion, acciones, cargar = False,nombre="elite.p"):
    poblacion = []
    #hacemos el calculo de la longitud del individuo
    n = 2173600 + (acciones*512) + acciones
    for i in range(numero_poblacion):
        
        if random.random() > .5 and cargar:
            poblacion.append(cargar_ultimo_elite(nombre))    
        else:
            individuo = pso.IndividuoPSO(n,acciones)
            poblacion.append((individuo,0))
    return poblacion
        