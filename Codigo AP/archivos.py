#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:55:41 2019

@author: yoshio
"""

# =============================================================================
# Métodos que nos permite cargar y guardar un objeto guardado, en este caso el elite
# Esto es usado durante el entrenamiento
# =============================================================================

#importación de la biblioteca de manejo de archivos
import pickle


# =============================================================================
# Método que nos permite guardar un elite en un archivo de texto
# =============================================================================
def guardar_pesos(pesos, nombre = "elite.p"):
    with open(nombre,"wb") as f:
        pickle.dump(pesos,f)
        
        
# =============================================================================
# Método que permite cargar el ultimo elite que se obtuvo
# =============================================================================
def cargar_ultimo_elite(nombre = "elite.p"):
    elite = None
    with open(nombre,"rb") as f:        
        elite = pickle.load(f)        
    return elite