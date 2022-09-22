#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esse código simula-se a dinâmica da equação diferencial não linear obtida pela modelagem caixa branca
para o sistema de escoamento de fluido, através do uso da biblioteca: Python Control Systems. 
Para tal, serão utilizados dois comando específicos, sendo eles: NonlinearIOSystem, input_output_response.

Para maiores informações consultem as instruções de uso das funções na documentação da
biblioteca.

Além disso, lembrem-se que nessa simulação assume-se que o fluido no interior do tanque é a água.

@author: Lucas Silva de Oliveira 
@data: 04/09/2022
"""
import numpy as np # importando biblioteca numpy
import matplotlib.pyplot as plt # importando biblioteca para plotar as figuras
import control as ct  #importanto biblioteca control

plt.close('all') #comando para fechar todas janelas de plot

#Criando um função para cálculo da equação diferencial do sistema------------------------------------
def model_update(t,x,u,params):
    # parametros do sistema
    r = 0.1   # raio do tanque cilindrico
    A = np.pi*r**2 # area
    beta = 0.0015     # coeficiente beta
    
    #Atenção: A vazão de entrada Q1 (Kg/S) é dada pela variável interna u, enquanto o estado do sistema, H,
    #é descrita na variável interna x. 
    if x<=0:
        x=0
    dh = u/A-(beta/A)*np.sqrt(x)
    
    return dh

def model_output(t,x,u,params):
    #Atenção os dados retornados nessa função estarão disponíveis para aquisição e tratamento externo. 
    #Para o caso em estudo, a saída do sistema, y, é o próprio estado, ou seja, y = H.
    
    return x

#Definindo o sistema não linear do tanque, segundo biblioteca----------------------------------------
#Observe a notação adotada para os seguintes parâmetros: name, inputs, outputs e states. Esses
#parâmetros serão utilizados para realização da conexão dos blocos da malha fechada, quando  necessário.
SYSTEM = ct.NonlinearIOSystem(model_update, model_output , states=1, name='SYSTEM',inputs = ('u'), outputs = ('y'))

#Definindo os parâmetros de simulação----------------------------------------------------------------
# tempo    
t0 = 0    # tempo inicial
tf = 150   # tempo final
t = np.linspace(t0,tf,100) # instantes que desejo ter a solucao

T = t[1]-t[0] #período de amostragem

u= 0.00094* np.ones(t.shape) #Sinal de controle necessário para levar o sistema para o 
#ponto de operação desejado em malha aberta.



X0 = 0.75 #Condição inicial do sistema.

#Executando a simulação do sistema não linear em malha aberta----------------------------------------
#Observe que para simulação em malha aberta a função exige os seguintes parâmetros:
#Sistema a ser simualado, vetor com o tempo de simulação, vetor sinal de controle, condição
#inicial do sistema.
t, y = ct.input_output_response(SYSTEM, t, u, X0) 

#Plotando o resultado da simulação-------------------------------------------------------------------
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,y,'b',label='h(t)')
plt.ylabel('h(t)[cm]')
plt.xlim(0,tf)
plt.legend()
plt.ylim(0,0.80)
plt.title('Resposta temporal do tanque em malha aberta')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,u,'b',label='u0')
plt.ylabel('u(t)[Kg/s]')
plt.legend()
plt.xlabel('Tempo [s]')
plt.xlim(0,tf)
plt.grid()
plt.show()




