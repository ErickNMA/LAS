# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:45:35 2022

@author: Lucas

Exemplo de simulação numérica integração de Euler
"""

# importa bibliotecas

import numpy as np
import matplotlib.pyplot as plt



plt.close('all') # Fecha gráficos

"""
Define EDO - Caso 2: tanque não linear

\dot{h}(t) = Fi(t)/A - (\beta/A)*\sqrt(h(t)) 

A = \pi r**2; r = 10cm
""" 

# Algumas soluções
# 1) vazão de entrada nula (Fi = 0) e condição inicial (h(0) = 0,75 m)):

# tempo    
t0 = 0    # tempo inicial
tf = 150   # tempo final
t = np.linspace(t0,tf,100) # instantes que desejo ter a solucao

T = t[1]-t[0] #período de amostragem

# parametros do sistema
r = 0.1   # raio do tanque cilindrico
A = np.pi*r**2 # area
beta = 0.0015     # coeficiente beta

#pre alocando os vetores
h = np.empty(len(t)) # alocando o vetor para a coluna de líquido
h.fill(np.nan) # limpando o vetor e definindo-o como NAN.

u = 0*np.ones(len(t)) # alocando o vetor para a coluna de líquido, 
#note que o vetor é multiplicado por 0, portanto torna-se nulo.


# condição inicial e valor de Fi (=u)
h[0] = 0.75 # nivel inicial


#integração via euler

for k in range(len(t)-1):
    h[k+1] = h[k]+ T*( u[k]/A-(beta/A)*np.sqrt(h[k]))
    if h[k+1]<=0:
        h[k+1] = 0

plt.figure(1)
plt.subplot(211) 
plt.plot(t, h, 'r',label='Nível, h(t) u = 0.0')

plt.subplot(212) 
plt.plot(t,u,'r',label='u=0')

# 2) vazão de entrada não nula (Fi = 0.00094) e 
#    condição inicial (h(0) = 0,75 m)):
    
#pre alocando os vetores
h2 = np.empty(len(t)) # alocando o vetor para a coluna de líquido
h2.fill(np.nan) # limpando o vetor e definindo-o como NAN.

u2 = 0.00094*np.ones(len(t)) # alocando o vetor para a coluna de líquido, 
#note que o vetor é multiplicado por 0, portanto torna-se nulo.


# condição inicial e valor de Fi (=u)
h2[0] = 0.75 # nivel inicial

#integração via euler

for k in range(len(t)-1):
    h2[k+1] = h2[k]+ T*( u2[k]/A-(beta/A)*np.sqrt(h2[k]))
    if h[k+1]<=0:
        h[k+1] = 0
plt.figure(1)
plt.subplot(211) 
plt.plot(t, h2, 'b',label='Nível, h(t) u = 0.00094')

plt.subplot(212) 
plt.plot(t,u2,'*b',label='u = 0.00094')

# 3) vazão de entrada não nula (Fi = 0.00094) e 
#    condição inicial nula (h(0) = 0 m)):

#pre alocando os vetores
h3 = np.empty(len(t)) # alocando o vetor para a coluna de líquido
h3.fill(np.nan) # limpando o vetor e definindo-o como NAN.

u3 = 0.00094*np.ones(len(t)) # alocando o vetor para a coluna de líquido, 
#note que o vetor é multiplicado por 0, portanto torna-se nulo.

# condição inicial e valor de Fi (=u)
h3[0] = 0 # nivel inicial

#integração via euler

for k in range(len(t)-1):
    h3[k+1] = h3[k]+ T*( u3[k]/A-(beta/A)*np.sqrt(h3[k]))
    if h[k+1]<=0:
        h[k+1] = 0
plt.figure(1)
plt.subplot(211) 
plt.plot(t, h3, 'm',label='Nível, h(t) u = 0.00094')
plt.ylabel('$h(t)$', fontsize=12)
plt.ylim(0,0.80)
plt.xlim(0,tf)
plt.grid()
plt.legend(fontsize=8)

plt.subplot(212) 
plt.plot(t,u3,'m',label='u = 0.00094')
plt.grid()
plt.ylabel('Fi ($m^3/s$)')
plt.xlabel('Tempo (s)')
plt.xlim(0,tf)