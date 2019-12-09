#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:13:16 2019

@author: yoshio
"""

# =============================================================================
# Importación de las bibliotecas necesarias
# =============================================================================
from tkinter import *
from tkinter import messagebox
import gym
import tensorflow as tf
import tkinter.ttk as ttk


#Sin esto no funciona
physical_devices = tf.config.experimental.list_physical_devices('CPU')

print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#cargamos el agente
import juego


# =============================================================================
# Método que nos permite iniciar con el entrenamiento y los datos configurados
# =============================================================================
def entrenamiento(*args):    
    if len(parametro_R.get()) == 0:
        messagebox.showwarning(message = 'Favor de llenar los campos correspondientes.\nRevise los campos para R, C1 y C2', title = 'Advertencia')
    else:
        
        mensaje = 'Se realizara el entrenamiento con los siguiente parametros.\n\nR  : '+parametro_R.get()+'\nC1 : '+parametro_c1.get()+'\nC2 : '+parametro_c2.get()+'\nFrames : '+numero_frame.get()+'\nPoblación : '+n_poblacion.get()+"\nJuego : "+juegos[seleccion_juego.get()-1]
        
        elite = False
        
        if int(cargar_elite.get()) == 1:
            elite = True
            mensaje = mensaje + '\nSe cargara el mejor elite guardado'
        
        messagebox.showinfo(message = mensaje, title = 'Entrenamiento')
                
        ambiente = juegos[seleccion_juego.get()-1] #Obtenemos el nombre del ambiente
        
        env      = gym.make(ambiente+"-v0") #Cargamos el ambiente          
        barra_progreso['value']  = 0
        barra_generacion['value']= 0
        
        if int(seleccion_entrenamiento.get()) == 1:       
            
            #Creamos nuestro diccionario de configuración
            configuracion = {'r'         : float(parametro_R.get()), 'c1'     : float(parametro_c1.get()), 
                             'c2'        : float(parametro_c2.get()), 'n'     : int(n_poblacion.get()),
                             'elite'     : elite,                     'frames':int(numero_frame.get()),
                             'visualizar': False,                  'instancia': infogeneracion, 
                             'env'       : env,                     'ambiente': ambiente,
                             'progreso'  : barra_progreso,        'generacion': barra_generacion,
                             'root'      : root}            
            _juego.iniciar_entrenamiento(configuracion)
        else:            
            #Creamos nuestro diccionario de configuración
            configuracion = {'n'     : int(n_poblacion.get()),
                             'elite'     : elite,                     'frames':int(numero_frame.get()),
                             'visualizar': False,                  'instancia': infogeneracion, 
                             'env'       : env,                     'ambiente': ambiente,
                             'progreso'  : barra_progreso,        'generacion': barra_generacion,
                             'root'      : root}    
            _juego.iniciar_entrenamiento_genetico(configuracion)
        



# =============================================================================
# Método que nos permite ejecutar el modelo entrenado
# =============================================================================
def ejecutar_prueba(*args):
    ambiente = juegos[seleccion_juego.get()-1] #Obtenemos el nombre del ambiente
    env      = gym.make(ambiente+"-v0") #Cargamos el ambiente        
    _juego.probar_modelo_entrenado(env,ambiente,int(seleccion_entrenamiento.get())) #enviamos el ambiente y el nomb


# =============================================================================
# Método que permite obtener el puntaje final del juego
# =============================================================================
def obtener_puntaje_registro(*args):
    barra_progreso['value']  = 0
    barra_generacion['value']= 0
        
    ambiente = juegos[seleccion_juego.get()-1]
    env = gym.make(ambiente+"-v0")
    configuracion = {'entorno' :  env, 'progreso'  : barra_progreso, 'root' : root}
    _juego.obtener_puntaje_modelo(configuracion,ambiente,int(seleccion_entrenamiento.get()))


#vector de nombres de los juegos
juegos = ["Asterix","Asteroids","Atlantis","Enduro","Frostbite","Gravitar","Seaquest","Kangaroo","Venture","Zaxxon"]      

#iniciamos con nuestro objeto padre grafico
root   = Tk()

#le ponemos el titulo
root.title("Aprendizaje profundo, Proyecto final")

# =============================================================================
# Creamos el frame raiz de todos
# =============================================================================
mainframe = ttk.Frame(root,padding="5 5 5 5")
mainframe.grid(column = 0, row = 0, sticky =(N,W,E,S))

# =============================================================================
# Creamos el frame de la configuración del entrenamiento, (1,1)
# =============================================================================
frameconfiguracion = ttk.Frame(mainframe) 
frameconfiguracion.grid(column = 1, row = 1, sticky = W, pady = "5 0 ")

#agregamos una etiqueda indicando el area de configuración
ttk.Label(frameconfiguracion, text = "------Configuración del entrenamiento------").grid(column = 1, row = 1, sticky=W)

# =============================================================================
# Insertamos las etiquetas de R, C1 y C2 en el frame
# =============================================================================
ttk.Label(frameconfiguracion, text = "R :").grid(column = 1, row = 2, sticky = W, padx = "50 0")
ttk.Label(frameconfiguracion, text = "C1:").grid(column = 1, row = 3, sticky = W, padx = "50 0")
ttk.Label(frameconfiguracion, text = "C2:").grid(column = 1, row = 4, sticky = W, padx = "50 0")
ttk.Label(frameconfiguracion, text = "frames:").grid(column = 1, row = 5, sticky = W, padx = "50 0")
ttk.Label(frameconfiguracion, text = "población:").grid(column = 1, row = 6, sticky = W, padx = "50 0")


# =============================================================================
# Creación de los parametros de entrada
# =============================================================================
parametro_R  = StringVar() #inidica el parametro R
parametro_c1 = StringVar() #indica el parametro C1
parametro_c2 = StringVar() #indica el parametro C2
cargar_elite = StringVar() #indica si podemos cargar el elite
numero_frame = StringVar() #Indica el número de frames para el agente
n_poblacion  = StringVar() #Indica el número de individuos en la población 


seleccion_juego = IntVar() #Indica que juego vamos a entrenar o ejecutar
seleccion_entrenamiento = IntVar()

# =============================================================================
# Parametros por default
# =============================================================================
parametro_R.set(0.9)
parametro_c1.set(1.6)
parametro_c2.set(1.7)
numero_frame.set(20000)
n_poblacion.set(100)
cargar_elite.set(0)        #establecemos el valor de cero


seleccion_juego.set(1)
seleccion_entrenamiento.set(1)

# =============================================================================
#  Creamos la entrada de texto para el parametro R
# =============================================================================
entradaR = ttk.Entry(frameconfiguracion, width = 5, textvariable = parametro_R)
entradaR.grid(column = 2, row = 2, sticky = E, pady = "5 5")

ttk.Label(frameconfiguracion, text ="---Juego entrenar y/o probar---").grid(column = 3, row = 1, sticky = E, pady ="5 5")

ttk.Radiobutton(frameconfiguracion, text="Asterix",   variable = seleccion_juego, value = 1).grid(column = 3, row = 2, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Asteroids", variable = seleccion_juego, value = 2).grid(column = 3, row = 3, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Atlantis",  variable = seleccion_juego, value = 3).grid(column = 3, row = 4, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Enduro",    variable = seleccion_juego, value = 4).grid(column = 3, row = 5, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Frostbite", variable = seleccion_juego, value = 5).grid(column = 3, row = 6, sticky = W, padx = "5")

ttk.Radiobutton(frameconfiguracion, text="Gravitar",  variable = seleccion_juego, value = 6).grid(column = 4, row = 2, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Seaquest",  variable = seleccion_juego, value = 7).grid(column = 4, row = 3, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Kangaroo",  variable = seleccion_juego, value = 8).grid(column = 4, row = 4, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Venture",   variable = seleccion_juego, value = 9).grid(column = 4, row = 5, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Zaxxon",    variable = seleccion_juego, value =10).grid(column = 4, row = 6, sticky = W, padx = "5")

ttk.Label(frameconfiguracion,text="--entrenamiento--").grid(column = 5, row = 1, sticky = E, pady = "5 5")

ttk.Radiobutton(frameconfiguracion, text="PSO",     variable = seleccion_entrenamiento, value =1).grid(column = 5, row = 2, sticky = W, padx = "5")
ttk.Radiobutton(frameconfiguracion, text="Genético",variable = seleccion_entrenamiento, value =2).grid(column = 5, row = 3, sticky = W, padx = "5")


# =============================================================================
# Creamos check para elegir si cargar el ultimo elite, o empezar desde cero
# =============================================================================
check_elite = ttk.Checkbutton(frameconfiguracion, text = "cargar archivo elite", variable = cargar_elite)
check_elite.grid(column = 2, row = 7, sticky = W, padx = "5 0")


# =============================================================================
# Creamos la entrada de texto para el parametro C1
# =============================================================================
entradac1 = ttk.Entry(frameconfiguracion, width = 5, textvariable = parametro_c1)
entradac1.grid(column = 2, row = 3, sticky = E, pady = "0 5")


# =============================================================================
# Creamos la entrada de texto para el parametro C2
# =============================================================================
entradac2 = ttk.Entry(frameconfiguracion, width = 5, textvariable = parametro_c2)
entradac2.grid(column = 2, row = 4, sticky = E, pady = "0 5")


# =============================================================================
# Creamos la entreda para los frames por individuo
# =============================================================================
entradaframe = ttk.Entry(frameconfiguracion,width = 5, textvariable = numero_frame) 
entradaframe.grid(column = 2, row = 5, sticky = E, pady = "0 5")


# =============================================================================
# creamos la entrada para la cantidad de población
# =============================================================================
entradapoblacion = ttk.Entry(frameconfiguracion, width = 5, textvariable = n_poblacion)
entradapoblacion.grid(column = 2, row = 6, sticky = E, pady = "0 5")


# =============================================================================
# Creamos el boton para entrenar la red
# =============================================================================
button = ttk.Button(frameconfiguracion, text = "Entrenar...", command = entrenamiento)
button.grid(column = 1, row = 7, sticky = W, pady = "5 0")


# =============================================================================
# Creamos el frame para las pruebas del entrenamiento (1,2)
# =============================================================================
frameseguimiento = ttk.Frame(mainframe)
frameseguimiento.grid(column = 1, row = 2, pady="5 0")

ttk.Label(frameseguimiento, text="---Visualización de entrenamiento, generaciones---").grid(column = 1, row = 1, sticky=W)

informacion_generaciones = StringVar()
infogeneracion = Text(frameseguimiento,width = 40, height = 10)
infogeneracion.grid(column = 1, row = 2, sticky = W)

ttk.Label(frameseguimiento,text= ' --progreso generación-- ').grid(column = 2, row = 1, sticky = W ) 
barra_generacion = ttk.Progressbar(frameseguimiento, orient = HORIZONTAL, length= 100, mode = 'determinate')
barra_generacion.grid(column = 2, row = 2)

ttk.Label(frameseguimiento,text = ' --progreso general--').grid(column = 3, row = 1, sticky = W)
barra_progreso = ttk.Progressbar(frameseguimiento, orient = HORIZONTAL, length= 100, mode = 'determinate')
barra_progreso.grid(column = 3, row = 2)

# =============================================================================
# creamos el frame de las validaciones (1,3)
# =============================================================================
framevalidacion = ttk.Frame(mainframe)
framevalidacion.grid(column = 1, row = 3, pady = "5 0")


#Creamos nuetro objeto del jueguillo
_juego = juego.agente(1000)

# =============================================================================
# Creamos el frame de pruebas (1,4)
# =============================================================================
framepruebas = ttk.Frame(mainframe)
framepruebas .grid(column = 1, row = 4, pady="5 0")


ttk.Label(framepruebas,text="----- probar modelo ----").grid(column = 1, row = 1, sticky=W )
ejecutar = ttk.Button(framepruebas, text='Ejecutar modelo entrenado...', command = ejecutar_prueba)
ejecutar.grid(column = 1, row = 2, pady = "5 0")

puntaje = ttk.Button(framepruebas, text='Puntaje final...', command = obtener_puntaje_registro)
puntaje.grid(column = 2, row = 2, pady = "5 0")

frameconfiguracion.focus()
root.resizable(width = False, height = False)
root.mainloop()