from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


print("Modelo de Regresion Lineal")
print("Probando la conexion con GitHub y Jenkins")
print("GitHub-Jenkins-Docker")

print("Actualizando direccion proporcionada por ngrok")



def suma(a, b):
    for n in (a,b):
        if not isinstance(n, int) and not isinstance(n, float):
            return TypeError
    return a + b


r = suma(2,2.2)
print("La suma es {}".format(r))
