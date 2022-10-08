# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')  #Se cierran pestañas

'''
Santiago Beltrán Jaramillo  (sbeltran@unal.edu.co)
https://github.com/Sbeltranj/elementos-finitos

El presente código toma referencias de:
    https://github.com/diegoandresalvarez
    https://github.com/michaelherediaperez
    Jorge Eduardo Hurtado Goméz (Análisis matricial de Estructuras)
    
##################################################################################

Análisis dinámico espectral, para un pórtico de 3 niveles  (H entrepiso 3.4 m); 
análizado para la dirección corta (Y), Piso1 D = 240 kgf/m2, Piso2 D = 240 kgf/m2,
Piso3 Dcubierta = 130 kgf/m2, vigas y columnas de igual dimensión. 
Y área de la losa = 260 m2.

Para el análisis espectral:
    Ciudad: Manizales
    Suelo: tipo D
    Importancia: I

'''
# se definen algunas constantes que hacen el código más legible
NL1, NL2 = 0, 1
X,   Y   = 0, 1

b = 0.35  #base del elemento
h = 0.35  #altura del elemnto
# %% defino las variables a utilizar
Aviga = b*h*1e2;    #se amplifica rigidez axial vigas (consideración de diafragma rígido (JEHG-Análisis de estrucutras))
Acol  = b*h      # m^2    área
Iviga = b*h**3/12;  Icol  = b*h**3/12  # m^4    inercia_y

#%% Se define la estructura
xnod = np.array([[0, 0],   # coordenadas de cada nodo [x, y]
                 [6, 0],
                 [12,0],
                 [0,3.4],
                 [6,3.4],
                 [12,3.4],
                 [0,6.8],
                 [6,6.8],
                 [12,6.8],
                 [0,10.2],
                 [6,10.2],
                 [12,10.2]])
                 
#Se definen las propiedades por elemnto
A     = [ Acol, Aviga, Acol, Aviga, Acol,Acol, Aviga, Acol, Aviga, Acol,Acol, Aviga, Acol, Aviga, Acol] # áreas
I     = [ Icol, Iviga, Icol, Iviga, Icol, Icol, Iviga, Icol, Iviga, Icol, Icol, Iviga, Icol, Iviga, Icol] 

# LaG: local a global: matriz que relaciona nodos locales y globales
# (se lee la barra x va del nodo i al nodo j)
LaG = np.array([[1, 4],  
                [4, 5], 
                [2, 5],
                [5, 6],   
                [3, 6],
                [4, 7],
                [7, 8],
                [5, 8],
                [8, 9],
                [6, 9],
                [7, 10],
                [10, 11],
                [8, 11],       # fila = barra
                [11, 12],      # col1 = nodo global asociado a nodo local 1
                [9, 12]]) - 1  # col2 = nodo global asociado a nodo local 2              

E  = 4700*(21)**0.5*1e3 # kN/m2

#Para la viga se sección cuadrada de Timoshenko-E
alpha = 5/6
Aast = Acol*alpha #área de cortnate ('método energías de deformación)
niu  = 0.2 # poisson
G = E/(2*(1+niu)) #Módulo de corte kN/m2

nno  = xnod.shape[0] # Número de nodos
nbar = LaG.shape[0]  # número de barras (número de filas de LaG)
ngdl = 3*nno         # número de grados de libertad (tres por nodo)

#%% gdl: grados de libertad
gdl = np.arange(ngdl).reshape(nno, 3)  # nodos vs grados de libertad

plt.figure(1)
for e in range(nbar):
   plt.plot(xnod[LaG[e,:],X], xnod[LaG[e,:],Y], 'b-')
   
   # Calculo la posición del centro de gravedad de la barra
   cgx = (xnod[LaG[e,NL1],X] + xnod[LaG[e,NL2],X])/2
   cgy = (xnod[LaG[e,NL1],Y] + xnod[LaG[e,NL2],Y])/2
   plt.text(cgx, cgy, str(e+1), color='red')

plt.plot(xnod[:,X], xnod[:,Y], 'ro')
for n in range(nno):
    plt.text(xnod[n,X], xnod[n,Y], str(n+1))
    
plt.axis('equal')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.title('Numeración de la estructura')
plt.show()

# %% separo memoria
K   = np.zeros((ngdl,ngdl))  
T   = nbar*[None]
L   = nbar*[None]
Ke  = nbar*[None]
idx = np.zeros((nbar,6), dtype=int)

for e in range(nbar):
   # saco los 6 gdls de la barra e
   idx[e] = np.r_[gdl[LaG[e,NL1],:], gdl[LaG[e,NL2],:]]
   
   # longitud de la barra
   x1, y1 = xnod[LaG[e,NL1], :]
   x2, y2 = xnod[LaG[e,NL2], :]
   
   
   # matriz de transformación Te
   L = np.hypot(x2-x1, y2-y1)    # longitud de la barra
   c = (x2-x1)/L; s = (y2-y1)/L  # coseno y seno de la inclinación
   T[e] =   np.array([[ c,  s,  0,  0,  0,  0],
                    [-s,  c,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0,  0],
                    [ 0,  0,  0,  c,  s,  0],
                    [ 0,  0,  0, -s,  c,  0],
                    [ 0,  0,  0,  0,  0,  1]])
   
   # vector fuerzas nodales equiv. y matriz de rigidez en coordenadas locales
   # incluye fuerzas por peso propio 
   AE = A[e]*E;       #L2=L**2
   EI = E*I[e];       #L3=L**3
    
    # para la matriz de Timoshenko
   beta = (12*EI)/(L**2*G*Aast)
   
   Keloc =  np.array([
                [  AE/L,                               0,                                                    0, -AE/L,                               0,                                                     0],
                [     0,  (12*EI)/(L*(L**2*beta + L**2)),                             (6*EI)/(L**2*(beta + 1)),     0, -(12*EI)/(L*(L**2*beta + L**2)),                              (6*EI)/(L**2*(beta + 1))],
                [     0,       (6*EI)/(L**2*beta + L**2),                         (EI*(beta + 4))/(L + L*beta),     0,      -(6*EI)/(L**2*beta + L**2),                         -(EI*(beta - 2))/(L + L*beta)],
                [ -AE/L,                               0,                                                    0,  AE/L,                               0,                                                     0],
                [     0, -(12*EI)/(L*(L**2*beta + L**2)),                            -(6*EI)/(L**2*(beta + 1)),     0,  (12*EI)/(L*(L**2*beta + L**2)),                             -(6*EI)/(L**2*(beta + 1))],
                [     0,       (6*EI)/(L**2*beta + L**2), (6*EI)/(L*(beta + 1)) - (EI*(beta + 4))/(L + L*beta),     0,      -(6*EI)/(L**2*beta + L**2), (6*EI)/(L*(beta + 1)) + (EI*(beta - 2))/(L + L*beta)]])
   
   # se convierten a coordenadas globales
   Ke[e] = T[e].T @ Keloc @ T[e]
   # se ensamblan
   K[np.ix_(idx[e],idx[e])] += Ke[e] # ensambla Ke{e} en K global

#%% Se calcula la Matriz condensada
p = np.array([10,19,28]) -1 # Primarios
s = np.setdiff1d(np.arange(9,ngdl,1), p) #secundarios

K0 = K[np.ix_(p,p)]
K1 = K[np.ix_(p,s)]
K2 = K[np.ix_(s,p)]
K3 = K[np.ix_(s,s)]
   
C = K0 - K1 @ np.linalg.inv(K3) @ K2 #Mtriz condensada por 1 pórtico

Kcon = C*4 # 4 porticos en dirección x
#%% Masa de piso
Mpiso1 = 260 #kgf/m2
Mpiso2 = 260 #kgf/m2
Mpiso3 = 130 #kgf/m2

conv1 = 9.81/1000 # 1 kgf a kN

# Matriz de masa.
M = np.diag([Mpiso1*conv1, Mpiso2*conv1, Mpiso3*conv1])*240/9.81   # [kn.s^2/m]
print('\nMatriz de Masa [kN.s^2/m]')
print(M)

#%% calculamos frecuencias resolviendo valores y vectores propios 
from scipy import linalg

omega2, vv = linalg.eigh(Kcon, M)  #omega2 == w2; vv == vectores modales

Npisos = 3

wwi = omega2**0.5      # [rad/s]   Vector de frecuencias angulares.       
T  = 2*np.pi/wwi       # [s]       Vector de periodos de la estructura.

tabla_vfp = pd.DataFrame(
    data = np.c_[T.round(5)],
    index = np.arange(Npisos)+1,
    columns = ["T [s]"]
)

print(f'\n{tabla_vfp}')

# Para matriz modal
Phi = np.zeros((Npisos, Npisos))      

# # eigh vs eig, eigh (Matriz simétrica, valores en orden )
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html

for e in range(Npisos):
    #Resulevo vectores propios con eigh  
    phi_n = vv[:,e]
    # Se normaliza el vector respecto a la masa 
    nom = phi_n.T @ M @ phi_n  # phiT*M*phi norma del vector
    # Se Añade el modo de vibración (vector) como columna en la matriz de modos Phi.
    # se divide sobre la raiz de la norma
    Phi[:, e] = phi_n/np.sqrt(nom)

#%%Comprobación 

p1 = Phi.T @ M @ Phi  # comprobación (Identidad)
p2 = Phi.T @ Kcon @ Phi #Segunda comprobación w2

alfa = Phi.T @ M @ np.ones((Npisos, 1))   # Cálculo de la participación modal
part_masa = alfa**2 / np.sum(alfa**2)    
por_part_masa = part_masa*100             # Porcentaje de participación.


#%% Análisis ADEE  Análisis Espectral Dinámico Elástico 
Sa = 0.8938 # % Aceleración de acuerdo a NSR-10 para la estrutura análizada

Rho = np.zeros((Npisos,Npisos)) #respuetas máxmias por piso Nabla-i
for e in range(3):
    Rho[e,e] = ((Sa*9.81*alfa[e])/omega2[e]) 

Dm = Phi@Rho #Los desplazamientos modales
Fm = Kcon@Dm #las fuerzas modales
V = np.zeros((Npisos,Npisos)) #Para los cortantes modales

# Se aplica el método SRSS para la combinación modal
Srss_D = np.zeros((Npisos,1)) #Para los desplzamiento
Srss_V = np.zeros((Npisos,1)) #Para los cortantes SRCS

# Se aplica el método SRSS para la combinación modal:
cuad = Dm**2 #cuadrado de los desplzamiento

#Cortantes: 
for i in range(Npisos):
    V[i,:] = Fm[i,:]+V[i-1,:]
    Srss_D[i,:] = sum(cuad[i,:])

cuadV = V**2 #cuadradod de los cortantes
for i in range(Npisos):
    Srss_V[i,:] = sum(cuadV[i,:])
    

Srss_D = Srss_D**0.5
Srss_V = Srss_V**0.5


# Desplazamientos en el empotramiento 
ac = np.append(Srss_D[::-1], 0)
# Calculo las derivas de piso
delta = ((np.flip(abs(np.diff(ac))))/3.4)*100
delta = np.flip((np.append(delta[::-1], 0))) #derivas

Srss_D = np.insert(Srss_D, 0, 0, axis=0)

print('\nFuerzas cortantes por piso kN:')
tabla_ = pd.DataFrame(
    data = np.c_[Srss_V.round(5)],
    index = np.arange(Npisos)+1,
    columns = ["kN"]
)

print(f'\n{tabla_}')


#%% Se gráfican los modos de vibrar 
#(se ha tomado de referencia: https://github.com/michaelherediaperez )

hh_piso = np.array([3.4, 2*3.4, 3*3.4]) 
# Organizo los arreglos para graficar.
grf_pisos = np.zeros(3+1); grf_pisos[1:]+=hh_piso
grf_Phi = np.zeros((3+1, 3)); grf_Phi[1:, :] += Phi

fig = plt.figure(2)
plt.plot(delta, grf_pisos, '*-b')
plt.plot(np.zeros(3+1), grf_pisos, '-k')
plt.plot(np.zeros(3+1), grf_pisos, 'or')
plt.grid(b=True, which='major', linestyle='-')
plt.xlabel('% Deriva en Y')
plt.ylabel('Altura [m] ')
plt.title('Derivas para la fuerza ADEE-Y %')

# -----------------------------------------------------------------------------
# Gráfico de los modos de vibración.
fig = plt.figure(3)
#fig.set_size_inches(10, 10)
plt.subplots_adjust(wspace=1, hspace=0.5)
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.grid(b=True, which='major', linestyle='-')
    ax.set_ylabel("10.2 [m]")
    # Gráfico de los modos.
    ax.plot(grf_Phi[:,i], grf_pisos, '*-b')      
    # Gráfico de referencia sistema simplificado.
    ax.plot(np.zeros(3+1), grf_pisos, '-k')
    ax.plot(np.zeros(3+1), grf_pisos, 'or')
    ax.set_title(f"Modo {i+1}: T{i+1} = {round(T[i],2)} [S]")
    # Ajusto los límites del eje x.
    ax.set_xlim([-0.5, 0.5])
plt.show()
# ende ende !!