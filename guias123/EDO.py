# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:45:35 2022

@author: valter

Exemplo de simulação numérica de EDO
"""

# importa bibliotecas

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp


plt.close('all') # Fecha gráficos

"""
Define EDO - Caso 2: tanque não linear

\dot{h}(t) = Fi(t)/A - (\beta/A)*\sqrt(h(t)) 

A = \pi r**2; r = 10cm
""" 

def dhdt(t,h,A,beta,Fi): # argumentos A, beta e Fi precisam ser informados!
    if h<0: h=0
    return Fi/A-(beta/A)*np.sqrt(h)    

# Algumas soluções
# 1) vazão de entrada nula (Fi = 0) e condição inicial (h(0) = 0,75 m)):

# tempo    
t0 = 0    # tempo inicial
tf = 150   # tempo final
t = np.linspace(t0,tf,100) # instantes que desejo ter a solulcao

# parametros do sistema
r = 0.1   # raio do tanque cilindrico
area = np.pi*r**2 # area
beta = 0.0015     # coeficiente beta

# condição inicial e valor de Fi (=u)
h0 = 0.75 # nivel inicial
u = 0 # valor de Fi -> sinal de controle, normalmente u!
sol1 = solve_ivp(dhdt, t_span=(t0,tf), y0=[h0], method='RK23',
                 t_eval=t, args=(area, beta, u))

h_sol1 = sol1.y[0]

plt.figure(1)
plt.plot(t, h_sol1, 'r',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.show()
plt.legend(fontsize=10)


# 2) vazão de entrada não nula (Fi = 0.00094) e 
#    condição inicial (h(0) = 0,75 m)):

u = 0.00094 # valor de Fi -> sinal de controle, normalmente u!
sol1 = solve_ivp(dhdt, t_span=(t0,tf), y0=[h0], method='RK23',
                 t_eval=t, args=(area, beta, u))

h_sol2 = sol1.y[0]

plt.figure(1)
plt.plot(t, h_sol2, 'b',label='Nível, h(t) u = 0.00094')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.show()
plt.legend(fontsize=10)


# 3) vazão de entrada não nula (Fi = 0.00094) e 
#    condição inicial nula (h(0) = 0 m)):

h0 = 0
u = 0.00094 # valor de Fi -> sinal de controle, normalmente u!
sol1 = solve_ivp(dhdt, t_span=(t0,tf), y0=[h0], method='RK23',
                 t_eval=t, args=(area, beta, u))

h_sol3 = sol1.y[0]

plt.figure(1)
plt.plot(t, h_sol3, 'm',label='Nível, h(t) u = 0.00094')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.xlim(0,tf)
plt.ylim(0,0.80)
plt.show()
plt.legend(fontsize=10)
