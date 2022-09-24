#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esse código simula-se a dinâmica da equação diferencial não linear obtida pela modelagem caixa branca
para o sistema de escoamento de fluido, através do uso da biblioteca: Python Control Systems. 
Para tal, serão utilizados dois comando específicos, sendo eles: NonlinearIOSystem, input_output_response.

Para maiores informações consultem as instruções de uso das funções na documentação da
biblioteca.

Além disso, lembrem-se que nessa simulação assume-se que o fluido no interior do tanque é a água.

@author: Erick Nathan M. Alves & Victor Sidnei Cotta
@data: 24/09/2022
"""
from lib2to3.pgen2.literals import evalString
import numpy as np # importando biblioteca numpy
import matplotlib.pyplot as plt # importando biblioteca para plotar as figuras
import control as ct  #importanto biblioteca control

plt.close('all') #comando para fechar todas janelas de plot

#Parâmetros do sistema:
Teq = 80
R = 100

#Criando um função para cálculo da equação diferencial do sistema------------------------------------
def model_update(t,x,u,params):    

    return ((-(x**(3/4))+(u*R*0.53025*(x**(0.0315/8.5))))/13.76023)

def model_output(t,x,u,params):
    #Atenção os dados retornados nessa função estarão disponíveis para aquisição e tratamento externo. 
    #Para o caso em estudo, a saída do sistema, y, é o próprio estado, ou seja, y = H.
    
    return x

#Definindo o sistema não linear do tanque, segundo biblioteca----------------------------------------
#Observe a notação adotada para os seguintes parâmetros: name, inputs, outputs e states. Esses
#parâmetros serão utilizados para realização da conexão dos blocos da malha fechada, quando  necessário.
SYSTEM = ct.NonlinearIOSystem(model_update, model_output , states=1, name='SYSTEM',inputs = ('u'), outputs = ('y'))

#Função para cálculo do sinal de controle dado um ponto de equilíbrio:
def sinal(eq):
    return ((eq**(3/4))/(R*0.53025*(eq**(0.0315/8.5))))

#Função para gerar o degrau:
def deg(array, value, t0, duration, step):
    for j in range(int(round((duration/step), 0))):
        i = int(round(((t0/step) + j), 0))
        array[i] = value
    return array

#Definindo os parâmetros de simulação----------------------------------------------------------------
T = 0.1 #período de amostragem
# tempo    
t0 = 0    # tempo inicial
tf = int(3200)   # tempo final
t = np.linspace(t0,tf,int((tf-t0)/T)) # instantes que desejo ter a solucao

#O sinal é calculado para Taq = Teq e Taq' = 0:
u = sinal(Teq) * np.ones(t.shape) #Sinal de controle necessário para levar o sistema para o 
#ponto de operação desejado em malha aberta.

#Aplicando os degraus:
tdeg = 400
degraus = [[5, 400], [-3, 800], [3, 1600], [-5, 2400]]
for i in range(len(degraus)):
    u = deg(u, sinal(Teq+degraus[i][0]), degraus[i][1], tdeg, T)

T0 = 27 #Condição inicial do sistema.

#Executando a simulação do sistema não linear em malha aberta----------------------------------------
#Observe que para simulação em malha aberta a função exige os seguintes parâmetros:
#Sistema a ser simualado, vetor com o tempo de simulação, vetor sinal de controle, condição
#inicial do sistema.
t, y = ct.input_output_response(SYSTEM, t, u, T0)

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,y,'r',label='Taq(t)')
plt.ylabel('Temperatura (°C)')
plt.xlim(0,tf)
plt.legend()
plt.ylim(20, 90)
plt.title('Resposta temporal do sistema em malha aberta')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,u,'b',label='u(t)')
plt.ylabel('Corrente Elétrica (A)')
plt.legend()
plt.xlabel('Tempo (s)')
plt.xlim(0,tf)
plt.grid()
plt.show()





#Resposta ao degrau com 3 parâmetros:

#Degrau (+5):
#Equilíbrio:
y0 = (Teq*np.ones(t.shape))
#Reta tangente:
y1 = (0.06*t + 54.83)
A1 = ((80-54.83)/0.06)
#Reta 0,63K:
y2 = ((80+(0.63*5))*np.ones(t.shape))
B1 = (((80+(0.63*5))-54.83)/0.06)
#Reta K:
y3 = (85*np.ones(t.shape))
C1 = ((85-54.83)/0.06)

print("\n=> Resposta ao degrau (+5°): \tA = " + str(round(A1, 4)) + "\t B = " + str(round(B1, 4)) + "\t C = " + str(round(C1, 4)))

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(t,y0,'black',label='Equilíbrio')
plt.plot(t,y1,'pink',label='Tangente')
plt.plot(t,y2,'yellow',ls='--',label='0,63k')
plt.plot(t,y3,'blue',ls='--',label='k')
plt.scatter(A1, 80, c='orange', linewidths=3, label='A')
plt.scatter(B1, (80+(0.63*5)), c='purple', linewidths=3, label='B')
plt.scatter(C1, 85, c='green', linewidths=3, label='C')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(400, 530)
plt.legend()
plt.ylim(79, 86)
plt.title('Resposta ao degrau com 3 parâmetros (+5°)')
plt.grid()

#Degrau (-5):
#Reta tangente:
y4 = (-0.08*t + 272.32)
A2 = ((80-272.32)/-0.08)
#Reta 0,63K:
y5 = ((80-(0.63*5))*np.ones(t.shape))
B2 = (((80-(0.63*5))-272.32)/-0.08)
#Reta K:
y6 = (75*np.ones(t.shape))
C2 = ((75-272.32)/-0.08)

print("\n=> Resposta ao degrau (-5°): \tA = " + str(round(A2, 4)) + "\t B = " + str(round(B2, 4)) + "\t C = " + str(round(C2, 4)))

plt.subplot(1, 2, 2)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(t,y0,'black',label='Equilíbrio')
plt.plot(t,y4,'pink',label='Tangente')
plt.plot(t,y5,'yellow',ls='--',label='0,63k')
plt.plot(t,y6,'blue',ls='--',label='k')
plt.scatter(A2, 80, c='orange', linewidths=3, label='A')
plt.scatter(B2, (80-(0.63*5)), c='purple', linewidths=3, label='B')
plt.scatter(C2, 75, c='green', linewidths=3, label='C')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(2400, 2520)
plt.legend()
plt.ylim(74, 81)
plt.title('Resposta ao degrau com 3 parâmetros (-5°)')
plt.grid()

plt.show()



#Comparando os resultados:
#Resposta ao degrau:
def s(t, k, tau, theta):
    return (k*(1-np.exp((theta-t)/tau)))

z_up = []
z_down = []
m_up = []
m_down = []
ttrunc = []

for i in t:
    if(i>=400):
        if((i>=400)and(i<800)):
            base = Teq
            j = 400
            k = 5
        elif((i>=800)and(i<1200)):
            base = Teq+5
            j = 800
            k = -8
        elif((i>=1200)and(i<1600)):
            base = Teq-3
            j = 1200
            k = 3
        elif((i>=1600)and(i<2000)):
            base = Teq
            j = 1600
            k = 3
        elif((i>=2000)and(i<2400)):
            base = Teq+3
            j = 2000
            k = -3
        elif((i>=2400)and(i<2800)):
            base = Teq
            j = 2400
            k = -5
        elif((i>=2800)and(i<3200)):
            base = Teq-5
            j = 2800
            k = 5
        
        z_up.append(base + s((i-j), k, 83.33, 19.5))
        z_down.append(base + s((i-j), k, 62.5, 4))
        m_up.append(base + s((i-j), k, 52.5, 19.5))
        m_down.append(base + s((i-j), k, 39.375, 4))
        ttrunc.append(i)
    




#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(3)
plt.subplot(2, 2, 1)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(ttrunc, m_up,'green',label='Miller +')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(350,tf)
plt.legend()
plt.ylim(70, 90)
plt.title('Miller +')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(ttrunc, m_down,'blue',label='Miller -')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(350,tf)
plt.legend()
plt.ylim(70, 90)
plt.title('Miller -')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(ttrunc, z_up,'yellow',label='Ziegler +')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(350,tf)
plt.legend()
plt.ylim(70, 90)
plt.title('Ziegler +')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(ttrunc, z_down,'orange',label='Ziegler -')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(350,tf)
plt.legend()
plt.ylim(70, 90)
plt.title('Ziegler -')
plt.grid()

plt.show()

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(4)
plt.plot(t,y,'red',label='Taq(t)')
plt.plot(ttrunc, m_up,'green',label='Miller +')
plt.plot(ttrunc, m_down,'blue',label='Miller -')
plt.plot(ttrunc, z_up,'yellow',label='Ziegler +')
plt.plot(ttrunc, z_down,'orange',label='Ziegler -')
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tempo (s)')
plt.xlim(350,tf)
plt.legend()
plt.ylim(70, 90)
plt.title('Comparação entre os modelos desenvolvidos')
plt.grid()
plt.show()