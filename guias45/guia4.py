#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Erick Nathan M. Alves & Victor Sidnei Cotta
@data: 28/09/2022
"""
from turtle import pos
import numpy as np # importando biblioteca numpy
import matplotlib.pyplot as plt # importando biblioteca para plotar as figuras
import control as ct  #importanto biblioteca control

plt.close('all') #comando para fechar todas janelas de plot





#Letra b)

#Definindo a função de transferência do sistema:
s = ct.tf('s')
G = ((s+3)/((s**2)+(6*s)+5))

#Convertendo para espaço de estados:
ss = ct.tf2ss(G)

#Convertendo para forma canônica controlável:
cc_ss = ct.canonical_form(ss, 'reachable')[0]

#Convertendo para forma canônica observável:
co_ss = ct.canonical_form(ss, 'observable')[0]

#Convertendo para forma canônica diagonal:
polos = np.linalg.eig(ct.ssdata(ss)[0])
A = np.diag([polos[0][0],polos[0][1]])
B = np.array([[1],[1]])
C = [(3+polos[0][0])/(polos[0][0]-polos[0][1]), (3+polos[0][1])/(polos[0][1]-polos[0][0])] #O número 3 é o termo b0 do numerador da FT
D = ct.ssdata(ss)[3]
cd_ss = ct.ss(A, B, C, D)

print("\n\n\n Forma canônica controlável: \n" + str(cc_ss))
print("\n\n\n Forma canônica observável: \n" + str(co_ss))
print("\n\n\n Forma canônica diagonal: \n" + str(cd_ss))





#Letra g)

# Resposta pelas equações de estado:
t = np.arange(0, 15, .5, dtype=float)
u = np.ones(t.shape)
Y0 =[0, 0]
def model_update(t, y, u, params):
    
    y_1 = y[0]
    y_2 = y[1]

    y1_dot = y_2
    y2_dot = -5*y_1 -6*y_2 + u

    return [y1_dot, y2_dot]
def model_output(t, y, u, params):
    return y
sys = ct.NonlinearIOSystem(model_update, model_output, states=2, name='sys', inputs=('u',), outputs=('y_1','y_2'))
t, ya = ct.input_output_response(sys, t, u, X0=Y0)

#Resposta pela matriz de transformação de estados encontrada:
yb = [[], []]
for i in range(len(t)):
    yb[0].append(((1/5) - ((1/4)*(np.exp(-t[i]))) + ((1/20)*(np.exp(-5*t[i])))))
    yb[1].append((1/4)*(np.exp(-t[i])) - (1/4)*(np.exp(-5*t[i])))

plt.figure(1)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.subplot(2, 1, 1)
plt.plot(t, ya[0],'b', label='$y_1(t)$', linewidth=3)
plt.plot(t, yb[0],'r--', label='$\phi_{(t)}$', linewidth=3)
plt.ylabel('$y_1$', fontsize=28)
plt.title('Resposta temporal do sistema com entrada em degrau', fontsize=28)
plt.legend(fontsize=18)
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t, ya[1],'y',label='$y_2(t)$', linewidth=3)
plt.plot(t, yb[1],'g--',label='$\phi(t)$', linewidth=3)
plt.ylabel('$y_2$', fontsize=28)
plt.xlabel('Tempo [s]', fontsize=28)
plt.legend(fontsize=18)
plt.grid()
plt.show()
plt.show()