import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#d_1=1.61
#d_2=1.55
#d_3=1.52
#d-4=1.61
#d_5=2.13
#d_6=3.92
#d_7=1.78
#d_8=1.53
#d_9=1.54
#d_10=1.56

#media_d_1=16.14
#media_d_2=15.52
#media_d_3=15.17
#media_d_4=8.04
#media_d_5=10.67
#media_d_6=19.62
#media_d_7=8.93
#media_d_8=7.66
#media_d_9=15.15
#media_d_10=15.63


d=np.array([47.8, 37.8, 27.8, 17.8, 7.8, 2.3, 12.3, 22.3, 32.3, 42.3])
sigma_d = np.full(d.shape, 0.1)
T = np.array([1.61, 1.55, 1.52, 1.61, 2.13, 3.92, 1.78, 1.53, 1.54, 1.56])
sigma_T = np.array([0.09, 0.03, 0.04, 0.07, 0.07, 0.02, 0.04, 0.05, 0.01, 0.03])


# Definizione dell’accelerazione di gravita‘.
g= 981
"""Modello per il periodo del pendolo.
"""
def period_model(z, l):
    return 2.0 * np.pi * np.sqrt(((l**2.0 / 12.0) + z**2.0) / (g * z))

plt.figure('Periodo')
# Scatter plot dei dati.
plt.scatter(d, T, marker='o', color ='chartreuse', label='Dati raccolti' )
plt.errorbar(d, T, sigma_T, sigma_d, fmt='|', label='Barre di errore' )
#troviamo i valori ottimali e la covarianza usando il fit del modello
popt, pcov = curve_fit(period_model, d, T, p0=[100], sigma=sigma_T)
l_hat = popt[0]
sigma_l = np.sqrt(pcov[0, 0])

# confronto tra best fit e misurazioni
print(l_hat, sigma_l)
# Grafico del modello di best-fit.
xx = np.linspace(1, 50,1000)
#facciamo il grafico
plt.plot(xx, period_model(xx, l_hat), label='Fit')
plt.xlabel('Distanza [cm]')
plt.ylabel('Periodo [s]')
plt.grid(which='both', ls='dashed', color='gray')
plt.legend()
plt.savefig('pendolo.pdf')


#grafico dei residui:
#definiamo cosa è lo scarto
r = T - period_model(d, l_hat)
print (r)

plt.figure('Scarto')

media_r = np.mean(r) #media degli scarti
print(media_r)

plt.errorbar(d, r, sigma_T,fmt='o',color ='blueviolet', ecolor='lime', label='Scarti ed errore di misura' )
plt.xlabel('Distanza [cm]')
plt.ylabel('Residui [s]')
plt.axhline(media_r, color='red', linestyle='--', label='Valore Medio degli scarti = %.2f'%media_r)
plt.axhline(0, color='gray')
plt.grid(which='both', ls='dashed', color='gray')
plt.legend()

plt.show()



