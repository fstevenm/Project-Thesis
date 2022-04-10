import numpy as np
import matplotlib.pyplot as plt

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

def rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,a,b,h,S_init,E_init,IA_init,IS_init,Z_init,Z0_init,warna):
    global sigma, Lambda, beta1, beta2, mu1, mu2, alpha, kappa, gamma1, gamma2, xi, p
    
    if warna == 'red':
        xi = 0.03
    elif warna == 'green':
        xi = 0.02
    elif warna == 'blue':
        xi = 0.01
    elif warna == 'orange':
        xi = 0.0
   
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
    p=0.2
    
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

    ''' Pilih plot '''
    plt.plot(t_s,IS_s, warna, label = '$\\xi = {0}$'.format(xi))
#    plt.plot(t_s,Z_s, warna , label = '$\\xi = {0}$'.format(xi))
#    plt.plot(t_s,Z0_s, warna , label = '$\\xi = {0}$'.format(xi))
    
    return t_s[-1]


rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,0,300,1,9999500,100,100,100,100,100,'red')
rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,0,300,1,9999500,100,100,100,100,100,'green')
rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,0,300,1,9999500,100,100,100,100,100,'blue')
rk4(dSdt,dEdt,dIAdt,dISdt,dZdt,dZ0dt,0,300,1,9999500,100,100,100,100,100,'orange')

plt.ticklabel_format(useOffset=False, style='plain')

''' Pilih title '''
plt.title('Pengaruh Parameter $\\xi$ terhadap Kelompok $Infected$ $Symptomatic$ $(I_S)$')
#plt.title('Pengaruh Parameter $\\xi$ terhadap Kelompok $Recovered$ $(Z)$')
#plt.title('Pengaruh Parameter $\\xi$ terhadap Kelompok $Susceptible$ $that$ $Previously$ $Infected$ $(Z_0)$')

plt.xlabel('$t$')
plt.ylabel('Jumlah individu')
plt.grid()
plt.axis([0,130,0,2000000]) 
#plt.axis([0,130,0,3500000]) 
#plt.axis([0,130,0,2000000]) 

plt.legend(loc = 'best')
figure = plt.gcf()
figure.set_size_inches(10, 6)
plt.savefig('Xi_IS.png',dpi=600)

plt.show()
