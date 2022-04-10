import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

sigma = 0
Lambda = (10**7)/(365*65)
beta1 = 1.727*(10**(-7))
beta2 = 7.478*(10**(-8))
mu1 = 1/(365*65)
mu2 = 0.082
alpha = 1/(5.2)
kappa = 0.19
gamma1 = 1/10
gamma2 = 1/14
xi = 0.02
p=0.2
    
def model1(P1,t):    
    S = P1[0]
    E = P1[1]
    IA = P1[2]
    IS = P1[3]
    Z = P1[4]
    Z0 = P1[5]

    dSdt = Lambda - (beta1*IA*S + beta2*IS*S) - sigma*S - mu1*S
    dEdt = (beta1*IA*S+beta2*IS*S) -alpha*E - mu2*E
    dIAdt = p*alpha*E - kappa*IA - gamma1*IA - mu2*IA
    dISdt = (1-p)*alpha*E - gamma2*IS + (beta1*Z0*IA+beta2*Z0*IS) + kappa*IA - mu2*IS
    dZdt = gamma1*IA + gamma2*IS + sigma*S - xi*Z - mu1*Z 
    dZ0dt = xi*Z - (beta1*Z0*IA+beta2*Z0*IS)- mu1*Z0
    
    return [dSdt, dEdt, dIAdt, dISdt, dZdt, dZ0dt]

def dSdt(t,S,IA,IS):
    return Lambda - (beta1*IA*S + beta2*IS*S) - sigma*S - mu1*S

def dEdt(t,S,E,IA,IS):
    return (beta1*IA*S+beta2*IS*S) -alpha*E - mu2*E
    
def dIAdt(t,E,IA):
    return p*alpha*E - kappa*IA - gamma1*IA - mu2*IA

def dISdt(t,E,IA,IS,Z0):
    return (1-p)*alpha*E - gamma2*IS + (beta1*Z0*IA+beta2*Z0*IS) + kappa*IA - mu2*IS

def dZdt(t,S,IA,IS,Z):
    return gamma1*IA + gamma2*IS + sigma*S - xi*Z - mu1*Z 

def dZ0dt(t,IA,IS,Z,Z0):
    return xi*Z - (beta1*Z0*IA+beta2*Z0*IS)- mu1*Z0

def rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,a,b,h,S_init,E_init,IA_init,IS_init,Z_init,Z0_init):
    
    n = int((b-a)/h + 1)
    
    t_s = a + np.arange(n)*h
    S_s = np.zeros(n)
    E_s = np.zeros(n)
    IA_s = np.zeros(n)
    IS_s = np.zeros(n)
    Z_s = np.zeros(n)
    Z0_s = np.zeros(n)
    
    S = S_init
    E = E_init
    IA = IA_init
    IS = IS_init
    Z = Z_init
    Z0 = Z0_init
    
    for j,t in enumerate(t_s):
        S_s[j] = S
        E_s[j] = E
        IA_s[j] = IA
        IS_s[j] = IS
        Z_s[j] = Z
        Z0_s[j] = Z0
        
        k1S = dSdt(t,S,IA,IS)
        k1E = dEdt(t,S,E,IA,IS)
        k1IA = dIAdt(t,E,IA)
        k1IS = dISdt(t,E,IA,IS,Z0)
        k1Z = dZdt(t,S,IA,IS,Z)
        k1Z0 = dZ0dt(t,IA,IS,Z,Z0)
        
        k2S = dSdt(t+(1/2)*h, S+(1/2)*k1S*h, IA+(1/2)*k1IA*h, IS+(1/2)*k1IS*h)
        k2E = dEdt(t+(1/2)*h, S+(1/2)*k1S*h, E+(1/2)*k1E*h, IA+(1/2)*k1IA*h, IS+(1/2)*k1IS*h)
        k2IA = dIAdt(t+(1/2)*h, E+(1/2)*k1E*h, IA+(1/2)*k1IA*h)
        k2IS = dISdt(t+(1/2)*h, E+(1/2)*k1E*h, IA+(1/2)*k1IA*h, IS+(1/2)*k1IS*h, Z0+(1/2)*k1Z0*h)
        k2Z = dZdt(t+(1/2)*h, S+(1/2)*k1S*h, IA+(1/2)*k1IA*h, IS+(1/2)*k1IS*h, Z+(1/2)*k1Z*h)
        k2Z0 = dZ0dt(t+(1/2)*h, IA+(1/2)*k1IA*h, IS+(1/2)*k1IS*h, Z+(1/2)*k1Z*h, Z0+(1/2)*k1Z0*h)
        
        k3S = dSdt(t+(1/2)*h, S+(1/2)*k2S*h, IA+(1/2)*k2IA*h, IS+(1/2)*k2IS*h)
        k3E = dEdt(t+(1/2)*h, S+(1/2)*k2S*h, E+(1/2)*k2E*h, IA+(1/2)*k2IA*h, IS+(1/2)*k2IS*h)
        k3IA = dIAdt(t+(1/2)*h, E+(1/2)*k2E*h, IA+(1/2)*k2IA*h)
        k3IS = dISdt(t+(1/2)*h, E+(1/2)*k2E*h, IA+(1/2)*k2IA*h, IS+(1/2)*k2IS*h, Z0+(1/2)*k2Z0*h)
        k3Z = dZdt(t+(1/2)*h, S+(1/2)*k2S*h, IA+(1/2)*k2IA*h, IS+(1/2)*k2IS*h, Z+(1/2)*k2Z*h)
        k3Z0 = dZ0dt(t+(1/2)*h, IA+(1/2)*k2IA*h, IS+(1/2)*k2IS*h, Z+(1/2)*k1Z*h, Z0+(1/2)*k2Z0*h)
        
        k4S = dSdt(t+h, S+k3S*h, IA+k3IA*h, IS+k3IS*h)
        k4E = dEdt(t+h, S+k3S*h, E+k3E*h, IA+k3IA*h, IS+k3IS*h)
        k4IA = dIAdt(t+h, E+k3E*h, IA+k3IA*h)
        k4IS = dISdt(t+h, E+k3E*h, IA+k3IA*h, IS+k3IS*h, Z0+k3Z0*h)
        k4Z = dZdt(t+h, S+k3S*h, IA+k3IA*h, IS+k3IS*h, Z+k3Z*h)
        k4Z0 = dZ0dt(t+h, IA+k3IA*h, IS+k3IS*h, Z+k3Z*h, Z0+k3Z0*h)
        
        S += (1/6)*(k1S+2*k2S+2*k3S+k4S)*h
        E += (1/6)*(k1E+2*k2E+2*k3E+k4E)*h
        IA += (1/6)*(k1IA+2*k2IA+2*k3IA+k4IA)*h
        IS += (1/6)*(k1IS+2*k2IS+2*k3IS+k4IS)*h
        Z += (1/6)*(k1Z+2*k2Z+2*k3Z+k4Z)*h
        Z0 += (1/6)*(k1Z0+2*k2Z0+2*k3Z0+k4Z0)*h

    '''Titik kesetimbangan'''
    t = np.arange(0,25000,1)
    S_tk=[]
    E_tk=[]
    IA_tk=[]
    IS_tk=[]
    Z_tk=[]
    Z0_tk=[]
    for i in range (len(t)):
        S_tk.append(2.967914167*10**6)
        E_tk.append(1080.537778)
        IA_tk.append(111.7181326)
        IS_tk.append(1890.543335)
        Z_tk.append(7295.156738)
        Z0_tk.append(1.027358884*10**6)
    
    '''Untuk melihat nilai saat t'''
#    for j in range(len(S_s)):
#        print('ke {0} \t'.format(j),S_s[j])
    
    plt.plot(t_s,S_s, 'blue', label = '$Susceptible$ $(S)$', linewidth = 2.5)
    plt.plot(t_s,E_s, 'red', label = '$Exposed$ $(E)$',linewidth = 2.5)
    plt.plot(t_s,IA_s, 'green', label = '$Infected$ $Asymptomatic$ $(I_A)$', linewidth = 2.5)
    plt.plot(t_s,IS_s, 'orange', label = '$Infected$ $Symptomatic$ $(I_S)$', linewidth = 2.5)
    plt.plot(t_s,Z_s, 'purple', label = '$Recovered$ $(Z)$', linewidth = 2.5)
    plt.plot(t_s,Z0_s, 'cyan', label = '$Susceptible$ $that$ $Previously$ $Infected$ $(Z_0)$', linewidth = 2.5)
    
    ''' Plot Odeint (disesuaikan untuk masing-masing kelompok)'''
    t = np.arange(0,25000,1)
    x0 = [10**7 , 100,100,100, 100,100]

    P1 = odeint(model1,x0,t)

    A1 = P1[:,0]
    B1 = P1[:,1]
    C1 = P1[:,2]
    D1 = P1[:,3]
    E1 = P1[:,4]
    F1 = P1[:,5]
    
    plt.plot(t,A1,'lime', linestyle = 'dashed', label = 'Hasil Odeint', linewidth = 2)
#    plt.plot(t,B1,'lime', linestyle = 'dashed', label = 'Hasil Odeint',linewidth = 2)
#    plt.plot(t,C1,'lime', linestyle = 'dashed', label = 'Hasil Odeint', linewidth = 2)
#    plt.plot(t,D1,'lime', linestyle = 'dashed', label = 'Hasil Odeint', linewidth = 2)
#    plt.plot(t,E1,'lime', linestyle = 'dashed', label = 'Hasil Odeint', linewidth = 2)
#    plt.plot(t,F1,'lime', linestyle = 'dashed', label = 'Hasil Odeint', linewidth = 2)
    
    '''Titik kesetimbangan disesuaikan untuk masing-masing kelompok '''
#    plt.plot(t,S_tk,'red', linestyle = 'dashed', label = 'Nilai Kesetimbangan', linewidth = 2)

    return t_s[-1]


''' Run RK4 '''
rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,0,25000,1,9999500,100,100,100,100,100)

''' Format Plot '''
plt.ticklabel_format(useOffset=False, style='plain')

plt.title('Penyelesaian Numeris')
plt.xlabel('$t$')
plt.ylabel('Jumlah individu')
plt.grid()
plt.axis([0,500,0,10000000]) #disesuaikan untuk masing-masing populasi

plt.legend(loc = 'best')
figure = plt.gcf()
figure.set_size_inches(10, 6)
plt.savefig('Semua.png',dpi=600) #save gambar

plt.show()