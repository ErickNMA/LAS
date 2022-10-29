#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Erick Nathan M. Alves & Victor Sidnei Cotta
@data: 29/10/2022
"""

#Importando as bibliotecas:
from operator import indexOf, mod
import numpy as np # importando biblioteca numpy
import matplotlib.pyplot as plt # importando biblioteca para plotar as figuras
import control as ct  #importando biblioteca control

plt.close('all') #comando para fechar todas janelas de plot





#Parâmetros do sistema:
La = .154
L1 = .155
Lt = .270
d = .02
m = .1
rho = 1.23
c = 2.05
mi = 5
g = 9.81

#Constantes do modelo:
K1 = ((d*rho*c*La*L1)/(2*m*(((Lt**2)/12)+(d**2))))
K2 = ((g*d)/(((Lt**2)/12)+(d**2)))
K3 = ((mi*d**2)/(m*(((Lt**2)/12)+(d**2))))

#Dinâmica do sistema:
X0 = [0, 0] #condições iniciais

#Caracterizando o modelo, em espaço de estados:
def model_update(t, x, u, params):
    
    x1 = x[0] # posicao
    x2 = x[1] # velocidade

    #Retorna as derivadas:
    return [x2, ((K1*(np.cos(x1)**2)*u[0]) - ((K2*np.sin(x1)) + (K3*x2)))]

#Função que retorna o estado:
def model_output(t, x, u, params):
    return x





print('\n\n\n**********************************************')

# (1) => Levar ao ponto de operação:

#Parâmetros de simulação:
t = np.arange(0, 5, .01) #criando array de tempo
u_eq = (((K2/K1)*np.sin(np.radians(50)))/(np.cos(np.radians(50))**2)) #sinal que leva ao ponto desejado
u = u_eq*np.ones(t.shape) #criando array com para o sinal

#Instanciando o sistema:
sys = ct.NonlinearIOSystem(model_update, model_output, states=2, name='sys', inputs=('u'), outputs=('x1','x2'))

#Solucionando o sistema em si:
t, x = ct.input_output_response(sys, t, u, X0=X0)

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t, np.degrees(x[0]),'r', label='$\theta(t)$')
plt.ylabel('$\\theta(t)$ [°]')
#plt.xlim(0, tf)
#plt.legend()
#plt.ylim(20, 90)
plt.title('Levando o Sitema ao Ponto de Equilíbrio')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, np.degrees(x[1]), 'b', label='$\omega(t)$')
plt.ylabel('$\omega(t)$ [°/s]')
#plt.legend()
#plt.xlim(0,tf)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, u, 'g', label='$u(t)$')
plt.ylabel('u(t)')
#plt.legend()
plt.xlabel('Tempo [s]')
#plt.xlim(0,tf)
plt.grid()
plt.show()



  

# (2) e (3) => Entrada em degrau:

#Parâmetros de simulação:
X0 = [np.radians(50), 0]
td = np.arange(0, 12, .01) #criando array de tempo
u = u_eq*np.ones(td.shape) #criando array com para o sinal
#Criação do degrau:
for i in range(len(td)):
    if(td[i] > 6):
        break
    if(td[i]>=1):
        u[i] = (u[i] + 0.2*u[i])    

#Solucionando o sistema em si:
td, xd = ct.input_output_response(sys, td, u, X0=X0)

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(td, np.degrees(xd[0]),'r', label='$\\theta(t)$')
plt.ylabel('$\\theta(t)$ [°]')
#plt.xlim(0, tf)
#plt.legend()
#plt.ylim(20, 90)
plt.title('Resposta à Entrada em Degrau')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(td, np.degrees(xd[1]), 'b', label='$\omega(t)$')
plt.ylabel('$\omega(t)$ [°/s]')
#plt.legend()
#plt.xlim(0,tf)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(td, u, 'g', label='$u_d(t)$')
plt.ylabel('$u_d(t)$')
#plt.legend()
plt.xlabel('Tempo [s]')
#plt.xlim(0,tf)
plt.grid()
plt.show()





# (5) => Obtenção dos parâmetros por análise gráfica:

Mp = (np.degrees(np.max(xd[0]))-53.13)
tp = (td[indexOf(xd[0], np.max(xd[0]))]-1)
zeta = (np.log(Mp)/np.sqrt((np.log(Mp)**2)+(np.pi**2)))
omega_d = (np.pi/tp)
omega_n = (omega_d/np.sqrt(1-(zeta**2)))

print("\n omega_n: " + str(round(omega_n, 4)))
print("\n zeta: " + str(round(zeta, 4)))
print("\n omega_d: " + str(round(omega_d, 4)))
print("\n overshoot: " + str(round(Mp, 4)))
print("\n tp: " + str(round(tp, 4)))





# (4) => Envoltórias:
#Ogata - 4Ed - pag.191
def es(t): #envoltória superior:
    return np.degrees(x[0][-1]+(np.exp(-zeta*omega_n*t)/np.sqrt(1-(zeta**2))))
def ei(t): #envoltória inferior:
    return np.degrees(x[0][-1]-(np.exp(-zeta*omega_n*t)/np.sqrt(1-(zeta**2))))

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(3)
plt.plot(t, np.degrees(x[0]),'b', label='$\\theta(t)$')
plt.plot(t, es(t),'r--', label='envoltórias')
plt.plot(t, ei(t),'r--')
plt.ylabel('$\\theta(t)$ [°]')
#plt.xlim(0, 5)
plt.legend()
plt.xlabel('Tempo [s]')
#plt.ylim(20, 90)
plt.title('Decaimento Exponencial Subamortecido')
plt.grid()
plt.show()





# (6) => Função de transferência do sistema:
K = (np.radians(53.15-50)/((1.2*u_eq)-u_eq)) #ganho estático do sistema
s = ct.tf('s')
G = ((K*(omega_n**2))/((s**2)+(2*zeta*omega_n*s)+(omega_n**2)))

# Simulando a função de transferência
td, xft = ct.forced_response(G, T=td, U=(u-u_eq))

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(4)
plt.plot(td, np.degrees(xd[0]),'b', label='EDO')
plt.plot(td, np.degrees(xft)+50,'r--', label='FT')
plt.ylabel('$\\theta(t)$ [°]')
#plt.xlim(0, 5)
plt.legend()
plt.xlabel('Tempo [s]')
#plt.ylim(20, 90)
plt.title('Simulação Temporal: EDO X FT')
plt.grid()
plt.show()





# (7) Validação da FT:
tv = np.arange(0, 25, .01) # vetor de tempo

# Criação dos degraus:
u1 = np.ones(500)*int(1.1*u_eq)
u2 = np.ones(500)*int(0.95*u_eq)
u3 = np.ones(500)*int(1.15*u_eq)
u4 = np.ones(500)*int(0.88*u_eq)
u5 = np.ones(500)*int(1.05*u_eq)
u = np.hstack((u1, u2, u3, u4, u5))

# Simulação via EDO:
tv, xve = ct.input_output_response(sys, tv, u, X0=X0)

# Simulação via FT:
tv, xvf = ct.forced_response(G, T=tv, U=u-u_eq)

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(5)
plt.plot(tv, np.degrees(xve[0]),'b', label='EDO')
plt.plot(tv, (np.degrees(xvf)+50),'r--', label='FT')
plt.ylabel('$\\theta(t)$ [°]')
#plt.xlim(0, 5)
plt.legend()
plt.xlabel('Tempo [s]')
#plt.ylim(20, 90)
plt.title('Validação do Modelo')
plt.grid()
plt.show()






# (8) Tempo de acomodação:

# Para 2%:
ls2 = (x[0][-1] + (0.02*x[0][-1])) # limite superior
li2 = (x[0][-1] - (0.02*x[0][-1])) # limite inferior

# Para 5%:
ls5 = (x[0][-1] + (0.05*x[0][-1])) # limite superior
li5 = (x[0][-1] - (0.05*x[0][-1])) # limite inferior

# Plotando ts no gráfico juntamente com o sinal subamortecido
plt.figure(6)
plt.plot(t, np.degrees(x[0]),color='r', label='$\\theta(t)$')
plt.axhline(np.degrees(ls2), color='b', label='$t_s$ (2%)', linestyle='--')
plt.axhline(np.degrees(li2), color='b', linestyle='--')
plt.axhline(np.degrees(ls5), color='g', label='$t_s$ (5%)', linestyle='--')
plt.axhline(np.degrees(li5), color='g', linestyle='--')
plt.grid()
plt.title('Critérios para o Tempo de Acomodação')
plt.xlabel('Tempo [s]')
plt.ylabel('$\\theta(t)$ [°]')
plt.legend()
plt.show()

# Calculando os tempos de acomodação para 2$ e 5%, respectivamente:
ts2 = (4/(zeta*omega_n))
ts5 = (3/(zeta*omega_n))

print("\n ts2: " + str(round(ts2, 4)))
print("\n ts5: " + str(round(ts5, 4)))





# (9) Raízes do polinômio característico:

polos = ct.poles(G)

print('\ngamma1: ' + str(np.round(polos[0], 4)))
print('\ngamma2: ' + str(np.round(polos[1], 4)))

# O polinômio característico possui raízes complexas conjugadas, o que caracteriza o comportamento oscilatório subamortecido.
# O sistema é estável devido a parte real das raízes ser negativa, o que assegura a convergência do sistema para um popnto de equilíbrio desejado.

print('\n**********************************************\n\n\n')