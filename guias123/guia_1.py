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
tf = 3200   # tempo final
t = np.linspace(t0,tf,int((tf-t0)/T)) # instantes que desejo ter a solucao

#O sinal é calculado para Taq = Teq e Taq' = 0:
u = sinal(Teq) * np.ones(t.shape) #Sinal de controle necessário para levar o sistema para o 
#ponto de operação desejado em malha aberta.

#Aplicando os degraus:
tdeg = 400
u = deg(u, sinal(Teq+5), 400, tdeg, T)
u = deg(u, sinal(Teq-3), 800, tdeg, T)
u = deg(u, sinal(Teq+3), 1600, tdeg, T)
u = deg(u, sinal(Teq-5), 2400, tdeg, T)

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
#plt.ylim(0,0.80)
plt.title('Resposta temporal do sistema em malha aberta')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,u,'b',label='u0')
plt.ylabel('Corrente Elétrica (A)')
plt.legend()
plt.xlabel('Tempo [s]')
plt.xlim(0,tf)
plt.grid()
plt.show()




