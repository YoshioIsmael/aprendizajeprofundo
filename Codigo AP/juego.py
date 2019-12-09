# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:32:17 2019

@author: yoshio
"""

# =============================================================================
# ~Importación de las bibliotecas necesarias
# =============================================================================
import tensorflow as tf
import tkinter.ttk as ttk
import RedNeuronal
import numpy as np
import IndividuoPSO as pso
import random as rd
import threading

from tkinter import *
from ServiciosPoblacion import *
from archivos import *
from Evaluador import *
from Subproceso import *

# =============================================================================
# ~~Clase que representa nuestro agente en un ambiente "estocastico"
# =============================================================================
class agente:
    
    
# =============================================================================
#     ~Constructor de la clase
# =============================================================================
    def __init__(self, puntos):
        pass                
                                
        
# =============================================================================
#     Método que nos permite obtener el mejor de la poblacion
# =============================================================================
    def __elegir_elite(self, individuos, grupo, frames, env,acciones,Pso=True): 
        
        #Ordenamos los individuos
        individuos.sort(key = lambda indi: indi[1])
        elites = []
        hilos  = []
        
        for i in range(1,grupo):
            elites.append(individuos[-i])                
            hilos.append(EvaluacionParalela(env,acciones,5,individuos[-i],frames,Name=str(i),pso=Pso))
            hilos[i-1].start()
            
        mt = threading.currentThread()
        for th in threading.enumerate():
            if th is mt:
                continue
            th.join() 
        
        for i in range(len(hilos)):            
            #Obtenemos los pesos para este individuo y su fitness
            a,b,_ = hilos[i].resultados()
            elites.append((a,b))
            
        elites.sort(key = lambda indi: indi[1])
        
        #se regresa el mejor calificado y los individuos ordenados
        return elites[-1],individuos        
    
 
                                        
# =============================================================================
#     Método que permite iniciar la red neuronal                
# =============================================================================
    def iniciar_configuracion_red_neuronal(self,acciones_posibles):
        self.red         = RedNeuronal.RedNeuronal(acciones_posibles)                #Creamos nuestra red neuronal  

    
               
        
# =============================================================================
#         Método que permite probar el modelo creado
# =============================================================================
    def probar_modelo_entrenado(self,env,entorno,tipo):
        
        #iniciamos nuestra red neuronal
        self.iniciar_configuracion_red_neuronal(env.action_space.n) #Iniciamos con la configuración de los objetos necesarios y la red
        
        pesos_actuales = None
        
        if tipo == 1:
            #cargamos el ultimo elite generado
            individuopso , _ = cargar_ultimo_elite(entorno+".p")         
            
            #Establecemos los pesos actuales
            pesos_actuales = individuopso.posicion
        else:
            pesos_actuales = cargar_ultimo_elite(entorno+"genetico.p")
            pesos_actuales = pesos_actuales[0:len(pesos_actuales)-1]
        
        #reiniciamos todo       
        meta        = False
        pesos_esta  = False        
        action      = 0
        iteraciones = 0
        meta        = False
        env.reset()
        
        while(not meta):            
                                 
            #en esta parte capturamos el frame
            frame_capturado,recompensa,meta,env = obtener_siguiente_estado(env,action)
            
            #se lo enviamos a la red
            resultado = self.red(frame_capturado)                     
                                
            if not pesos_esta:
                #se establecen a la red                
                self.red.establecer_pesos(convertir_vector_a_parametros(pesos_actuales, env.action_space.n))
                pesos_esta = True
                        
            action      = tf.keras.backend.eval(tf.math.argmax(resultado[0]))                         
            iteraciones += 1
                
            env.render()     
            
            if meta:
                break
            
        messagebox.showinfo(message = 'La prueba termino', title = 'Prueba')
        env.close()
        
        
# =============================================================================
#         Método que permite probar el modelo creado y obtener el puntaje final
# =============================================================================
    def obtener_puntaje_modelo(self,configuracion,entorno,tipo):
        
        env      = configuracion['entorno']
        progreso = configuracion['progreso']
        root     = configuracion['root']
        #iniciamos nuestra red neuronal
        self.iniciar_configuracion_red_neuronal(env.action_space.n) #Iniciamos con la configuración de los objetos necesarios y la red
        
        pesos_actuales = None
        
        if tipo == 1:
            #cargamos el ultimo elite generado
            individuopso , _ = cargar_ultimo_elite(entorno+".p")         
            
            #Establecemos los pesos actuales
            pesos_actuales = individuopso.posicion
        else:
            pesos_actuales = cargar_ultimo_elite(entorno+"genetico.p")
            pesos_actuales = pesos_actuales[0:len(pesos_actuales)-1]
        
        #reiniciamos todo       
        meta        = False
        pesos_esta  = False        
        action      = 0
        meta        = False
        
        corrida_total  = 200
        puntos_totales = 0
        
        for i in range(corrida_total):
            meta = False
            env.reset()
            while(not meta):            
                                     
                #en esta parte capturamos el frame
                frame_capturado,recompensa,meta,env = obtener_siguiente_estado(env,action)
                
                #se lo enviamos a la red
                resultado = self.red(frame_capturado)                     
                                    
                if not pesos_esta:
                    #se establecen a la red                
                    self.red.establecer_pesos(convertir_vector_a_parametros(pesos_actuales, env.action_space.n))
                    pesos_esta = True
                            
                action      = tf.keras.backend.eval(tf.math.argmax(resultado[0]))   
                
                puntos_totales += recompensa
            progreso['value']  = int(((i+1)*100)/corrida_total)
            root.update_idletasks() 
                          
            
        puntos_totales = puntos_totales/corrida_total
        messagebox.showinfo(message = 'La prueba termino con el total de puntos '+str(puntos_totales), title = 'Prueba final')
        env.close()            
    
    
    
# =============================================================================
#     ~Método que nos permite iniciar con la ejecución del juego
# =============================================================================
    def iniciar_entrenamiento(self,configuraciones):        
        
        #nuestro elite
        ELITE      = None     
        
        #Obtenemos el ambiente en el cual se ejecutara el agente
        entorno = configuraciones['ambiente']
        
        #Cargamos el ambiente de la biblioteca de gym
        env      = configuraciones['env']        
        progreso = configuraciones['progreso']
        pro_gen  = configuraciones['generacion']
        root     = configuraciones['root']
        
        self.iniciar_configuracion_red_neuronal(env.action_space.n) #Iniciamos con la configuración de los objetos necesarios y la red
        
        #parametros pso        
        R  = configuraciones['r']  #peso de innercia incial
        c1 = configuraciones['c1'] #coeficiente de aceleracion inicial
        c2 = configuraciones['c2'] #coeficiente de aceleracion final
        n  = configuraciones['n']  #obtenemos el tamaño de la población
        panel = configuraciones['instancia']
                
        mensajes = "entrenamiento iniciado..."
        panel.replace(1.0, END, mensajes)
        panel.update()        
        
        frames_totales  = 20000000                           #indicador del número de iteraciones            
        frames_actuales = 0
        individuos      = generar_poblacion_pso(n, env.action_space.n, cargar = configuraciones['elite'],nombre = entorno+".p") #Creamos los individuos de la población pso        
        generacion      = 0                
                
        #iteramos hasta que se nos acaben las fotos
        while frames_totales > frames_actuales:
            #incrementamos el contrador de las generaciones
            generacion       +=1     
            pro_gen['value'] = 0
            hilos            = []            
            root.update_idletasks()              
            
            for i in range(len(individuos)):    
                hilos.append(EvaluacionParalela(entorno+"-v0",env.action_space.n,1,individuos[i],configuraciones['frames'],Name=str(i)))
                hilos[i].start()
                
            mt = threading.currentThread()
            for th in threading.enumerate():
                if th is mt:
                    continue
                th.join() 
                
            for i in range(len(hilos)):
                objetopso, fitness = individuos[i]                
                _,puntos,frames_recorridos = hilos[i].resultados()
                
                if(puntos > objetopso.fitness):                        
                    objetopso.mejorPosicion = objetopso.posicion    
                        
                objetopso.fitness = puntos                
                individuos[i]     = (objetopso, puntos)
                frames_actuales   += frames_recorridos
                pro_gen['value']  = int(((i+1)*100)/n)
                root.update_idletasks() 
                
            #elegimos a la mejor particula, elegimos un grupo de elites de 5     
            (C,P),individuos = self.__elegir_elite(individuos,5, configuraciones['frames'],entorno+"-v0",env.action_space.n)            
            
            hilos = []
            
            #Realizamos una evaluación para el mejor elite, en 200 corridas independientes
            for i in range(200):
                hilos.append(EvaluacionParalela(entorno+"-v0",env.action_space.n,1,(C,P),configuraciones['frames'],Name=str(i)))
                hilos[i].start()
                
            mt = threading.currentThread()
            for th in threading.enumerate():
                if th is mt:
                    continue
                th.join()   
                
            P = 0
            #promediamos
            for i in range(len(hilos)):
                _,puntos,_ = hilos[i].resultados()                
                P += puntos
            
            P = P/200
            
            
            if ELITE == None :
                ELITE = C,P
                guardar_pesos(ELITE,entorno+".p") #guardamos el objeto completo, elite
                guardar_pesos(ELITE, nombre = entorno+str(generacion)+".p")                
                mensajes = mensajes + '\nMejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames'
                print('Mejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames')            
                panel.replace(1.0, END, mensajes)
                panel.update() 
            else:
                _,EC = ELITE
                
                if(P > EC):
                    ELITE = C,P
                    guardar_pesos(ELITE,entorno+".p") #guardamos el objeto completo, elite
                    guardar_pesos(ELITE, nombre = entorno+str(generacion)+".p")
                    mensajes = mensajes + '\nMejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames'
                    print('Mejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames')            
                    panel.replace(1.0, END, mensajes)
                    panel.update() 
                                    
            elite,_= ELITE            
            
            #mueve todas las particulas                    
            for i in range(len(individuos)):
                individ,_ = individuos[i]
                individ.mover_particula(elite.posicion,R,c1,c2)
            
            progreso['value'] = int( (100 * frames_actuales)/frames_totales)
                                    
        messagebox.showinfo(message = "El entrenamiento finalizo con "+str(generacion)+" generaciones", title = 'Entrenamiento')
        env.close()
        
        
        
# =============================================================================
#         Método que permite realizar un entrenamiento con algotimos genetico
# =============================================================================
    def iniciar_entrenamiento_genetico(self,configuraciones):        
        
        #nuestro elite
        ELITE = None      
        
        #Obtenemos el ambiente en el cual se ejecutara el agente
        entorno = configuraciones['ambiente']
        
        #Cargamos el ambiente de la biblioteca de gym
        env      = configuraciones['env']        
        progreso = configuraciones['progreso']
        pro_gen  = configuraciones['generacion']
        root     = configuraciones['root']
        n        = configuraciones['n']  #obtenemos el tamaño de la población
        t        = 10
        self.iniciar_configuracion_red_neuronal(env.action_space.n) #Iniciamos con la configuración de los objetos necesarios y la red
        
        #parametros pso        
        panel = configuraciones['instancia']
        
        mensajes = "entrenamiento iniciado..."
        panel.replace(1.0, END, mensajes)
        panel.update()        
        
        frames_totales  = 20000000                           #indicador del número de iteraciones            
        frames_actuales = 0
        individuos      = generar_poblacion_genetico(n, env.action_space.n,cargar = configuraciones['elite'],nombre = entorno+"genetico.p") #Creamos los individuos de la población pso        
        generacion      = -1                
        longitud_indivi = len(individuos[0])-1
        ELITE           = None
        
        
        #iteramos hasta que se nos acaben las fotos
        while frames_totales > frames_actuales:
            
            #incrementamos el contrador de las generaciones
            generacion       +=1     
            pro_gen['value'] = 0
            root.update_idletasks()
            hilos            = []
            
            for i in range(len(individuos)):                 #Iteramos por toda la poblacion completa                
                
                if generacion != 0:
                    individuos.sort(key = lambda indi: indi[1])                    
                    item = individuos[random.randrange(n-t,n)]
                    individuos[i] = mutar_individuo(item,0.002)
                    
                hilos.append(EvaluacionParalela(entorno+"-v0",env.action_space.n,1,individuos[i],configuraciones['frames'],Name=str(i),pso = False))
                hilos[i].start()    
                
            mt = threading.currentThread()
            for th in threading.enumerate():
                if th is mt:
                    continue
                th.join()
            
            for i in range(len(hilos)):
                item,puntos,frames_recorridos = hilos[i].resultados()
                individuos[i] = (item,puntos)
                pro_gen['value']  = int(((i+1)*100)/n)
                frames_actuales   += frames_recorridos
                root.update_idletasks()
                
            #elegimos a la mejor particula, elegimos un grupo de elites de 5     
            (C,P),individuos = self.__elegir_elite(individuos,5, configuraciones['frames'], entorno+"-v0", env.action_space.n, False)
            
            hilos = []
            
            for i in range(200):
                hilos.append(EvaluacionParalela(entorno+"-v0",env.action_space.n,1,(C,P),configuraciones['frames'],Name=str(i),pso = False))
                hilos[i].start()
                
            mt = threading.currentThread()
            for th in threading.enumerate():
                if th is mt:
                    continue
                th.join()                 
                
            P = 0
            
            for i in range(len(hilos)):
                _,puntos,_ = hilos[i].resultados()                
                P += puntos
            
            P = P/200
            
            if ELITE == None :
                ELITE = C,P
                guardar_pesos(ELITE,entorno+"genetico.p") #guardamos el objeto completo, elite
                guardar_pesos(ELITE, nombre = entorno+str(generacion)+"genetico.p")                
                mensajes = mensajes + '\nMejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames'
                print('Mejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames')            
                panel.replace(1.0, END, mensajes)
                panel.update() 
            else:
                _,EC = ELITE
                
                if(P > EC):
                    ELITE = C,P
                    guardar_pesos(ELITE,entorno+"genetico.p") #guardamos el objeto completo, elite
                    guardar_pesos(ELITE, nombre = entorno+str(generacion)+"genetico.p")
                    mensajes = mensajes + '\nMejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames'
                    print('Mejor individuo : '+str(P)+' y faltan '+str(frames_totales-frames_actuales)+' frames')            
                    panel.replace(1.0, END, mensajes)
                    panel.update() 
                                    
            elite,_= ELITE      
             
            individuos[0] = (elite, 0)
            progreso['value'] = int( (100 * frames_actuales)/frames_totales)
                                    
        messagebox.showinfo(message = "El entrenamiento finalizo con "+str(generacion)+" generaciones", title = 'Entrenamiento')
        env.close()
