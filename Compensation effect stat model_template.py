import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#imports constable data from previous experiments 
Filepath='#filepath'
headerlist=['Ea','lnA']
Cons_5cal=pd.read_csv(Filepath+'#Constable data 1', names=headerlist)
Cons_5uncal=pd.read_csv(Filepath+'#Constable data 2',  names=headerlist)
Cons_2cal=pd.read_csv(Filepath+'#Constable data 3',  names=headerlist)
Cons_2uncal=pd.read_csv(Filepath+'#Constable data 4',  names=headerlist)
ConstableData=pd.concat([Cons_5cal,Cons_5uncal,Cons_2cal,Cons_2uncal],ignore_index=True)

#Curve fitting using scipy of constable data
def objective(x,a,b,c):
    return a*x+b
popt,_=curve_fit(objective, ConstableData['Ea'],ConstableData['lnA'])
m,c,k=popt
y_pred=m*ConstableData['Ea']+c

#plots constable data and fit linear regression
plt.plot(ConstableData['Ea'],ConstableData['lnA'],'ro')
plt.plot(ConstableData['Ea'],y_pred,'b--', label='y = -3.063 + 1.235E-4 * x')
plt.xlabel('Ea (J/mol)')
plt.ylabel('lnA')
plt.title('Constable plot for NMC 111')
plt.legend()
plt.show()


#Least square estimators method of SLR and single factor ANOVA
n=16
xbar=(sum(ConstableData['Ea']))/len(ConstableData)
ybar=(sum(ConstableData['lnA']))/len(ConstableData)
Sxx=sum((ConstableData['Ea']-xbar)**2)
Sxy=sum((ConstableData['Ea']*ConstableData['lnA']))-((sum(ConstableData['Ea'])*sum(ConstableData['lnA']))/n)
#manually verifying accuracy of previous linear model
B1=Sxy/Sxx
B0=ybar-(xbar*B1)
SSErr=sum((ConstableData['lnA']-(B0+(B1*ConstableData['Ea'])))**2)
SSr=(B1*Sxy)
SSTot=SSErr+SSr
Sigma_sqr=SSErr/(n-2)
se_B1=np.sqrt((Sigma_sqr**2)/Sxx)
se_B0=np.sqrt((Sigma_sqr**2)*((1/n)+((xbar**2)/Sxx)))

#T test
Ttest1=abs(B1/(se_B1))
Ttest0=abs(B0/(se_B0))
T_table=1.761

#ANOVA-F test
MSe=SSErr/(n-2)
MSr=SSr/1
F0=MSr/MSe
Ftable=	4.6001
F_cumprob=0.997023
p_dist=1-F_cumprob

#R^2
R2=1-(SSr/SSTot)