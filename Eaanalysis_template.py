import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import diff
from scipy.optimize import curve_fit

#define file path where data is located
TGA_path='#filepath'

#define column names for dataframes
headerlist_1=["Time (min)", "Temp (C) 2c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 2c"]
headerlist_2=["Time (min)", "Temp (C) 4c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 4c"]
headerlist_3=["Time (min)", "Temp (C) 6c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 6c"]
headerlist_4=["Time (min)", "Temp (C) 8c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 8c"]

#use read_csv to import data in .csv format for each heating rate
#first 1000 columns are dropped as they are noise from TGA and rounds alpha data to 3 decimals
TGA_2c=pd.read_csv(TGA_path+'#heating rate 1',names=headerlist_1)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Weight (%)', 'Weight (mg)'], axis='columns')
TGA_2c=TGA_2c.drop(TGA_2c.index[0:1000]).astype(float).round(3)
TGA_4c=pd.read_csv(TGA_path+'#heating rate 2',names=headerlist_2)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Weight (%)', 'Weight (mg)'], axis='columns')
TGA_4c=TGA_4c.drop(TGA_4c.index[0:1000]).astype(float).round(3)
TGA_6c=pd.read_csv(TGA_path+'#heating rate 3',names=headerlist_3)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Weight (%)', 'Weight (mg)'], axis='columns')
TGA_6c=TGA_6c.drop(TGA_6c.index[0:1000]).astype(float).round(3)
TGA_8c=pd.read_csv(TGA_path+'#heating rate 4',names=headerlist_4)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Weight (%)', 'Weight (mg)'], axis='columns')
TGA_8c=TGA_8c.drop(TGA_8c.index[0:1000]).astype(float).round(3)

#combines data into single frame
FullFrame=pd.concat([TGA_2c,TGA_4c,TGA_6c,TGA_8c], axis=1)

#searches full dataframe for specified alpha values and returns full row where true
alpha1=[0.100,0.200,0.300,0.400,0.500,0.600,0.700,0.800,0.900]
Alpha1=FullFrame[FullFrame['alpha 2c'].isin(alpha1)]
Alpha1=Alpha1.drop(Alpha1.index[2:5])\
    .drop(['Temp (C) 8c','Temp (C) 4c','Temp (C) 6c', 'alpha 8c','alpha 4c','alpha 6c']\
      , axis='columns').reset_index().drop_duplicates(subset='alpha 2c',keep='first', ignore_index=True)
Alpha2=FullFrame[FullFrame['alpha 4c'].isin(alpha1)]
Alpha2=Alpha2.drop(Alpha2.index[2])\
    .drop(['Temp (C) 2c','Temp (C) 8c','Temp (C) 6c', 'alpha 2c','alpha 8c','alpha 6c'],\
      axis='columns').reset_index().drop_duplicates(subset='alpha 4c',keep='first', ignore_index=True)
Alpha3=FullFrame[FullFrame['alpha 6c'].isin(alpha1)]
Alpha3=Alpha3.drop(['Temp (C) 2c','Temp (C) 4c','Temp (C) 8c', 'alpha 2c','alpha 4c','alpha 8c'],\
                   axis='columns').reset_index().drop_duplicates(subset='alpha 6c',keep='first', ignore_index=True)
Alpha4=FullFrame[FullFrame['alpha 8c'].isin(alpha1)]
Alpha4=Alpha4.drop(['Temp (C) 2c','Temp (C) 4c','Temp (C) 6c', 'alpha 2c','alpha 4c','alpha 6c'],\
                   axis='columns').reset_index().drop_duplicates(subset='alpha 8c',keep='first', ignore_index=True)
#combines all these alpha rows and drops redundant index columns
ra=pd.concat([Alpha1,Alpha2,Alpha3,Alpha4], ignore_index=True, axis=1)
ra1=ra.drop([0,3,6,9],axis=1)


#creates independent temperature variable for each alpha 
T01=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[1:9],axis=0).to_numpy()
T02=ra1.drop([2,5,8,11],axis=1).drop([0], axis=0).drop(ra1.index[2:9],axis=0).to_numpy()
T03=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:2], axis=0).drop(ra1.index[3:9],axis=0).to_numpy()
T04=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:3], axis=0).drop(ra1.index[4:9],axis=0).to_numpy()
T05=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:4], axis=0).drop(ra1.index[5:9],axis=0).to_numpy()
T06=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:5], axis=0).drop(ra1.index[6:9],axis=0).to_numpy()
T07=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:6], axis=0).drop(ra1.index[7:9],axis=0).to_numpy()
T08=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:7], axis=0).drop(ra1.index[8:9],axis=0).to_numpy()
T09=ra1.drop([2,5,8,11],axis=1).drop(ra1.index[0:8], axis=0).to_numpy()

#inverse temperature in Kelvin
T1=((T01+273)**-1).ravel()
T2=((T02+273)**-1).ravel()
T3=((T03+273)**-1).ravel()
T4=((T04+273)**-1).ravel()
T5=((T05+273)**-1).ravel()
T6=((T06+273)**-1).ravel()
T7=((T07+273)**-1).ravel()
T8=((T08+273)**-1).ravel()
T9=((T09+273)**-1).ravel()
Blist=[2,4,6,8]
#heating rate as used in OFW method
B_ofw=np.log(Blist).ravel()
#heating rate as used in KAS method
B1=(np.log(Blist/((T01+273)**2))).ravel()
B2=(np.log(Blist/((T02+273)**2))).ravel()
B3=(np.log(Blist/((T03+273)**2))).ravel()
B4=(np.log(Blist/((T04+273)**2))).ravel()
B5=(np.log(Blist/((T05+273)**2))).ravel()
B6=(np.log(Blist/((T06+273)**2))).ravel()
B7=(np.log(Blist/((T07+273)**2))).ravel()
B8=(np.log(Blist/((T08+273)**2))).ravel()
B9=(np.log(Blist/((T09+273)**2))).ravel()
#Performs regression to calculate slope for KAS method
m1, c1= np.polyfit(T1, B1, 1)
m2, c2= np.polyfit(T2, B2, 1)
m3, c3= np.polyfit(T3, B3, 1)
m4, c4= np.polyfit(T4, B4, 1)
m5, c5= np.polyfit(T5, B5, 1)
m6, c6= np.polyfit(T6, B6, 1)
m7, c7= np.polyfit(T7, B7, 1)
m8, c8= np.polyfit(T8, B8, 1)
m9, c9= np.polyfit(T9, B9, 1)
#Performs regression to calculate slope for OFW method
m01, c1= np.polyfit(T1, B_ofw, 1)
m02, c2= np.polyfit(T2, B_ofw, 1)
m03, c3= np.polyfit(T3, B_ofw, 1)
m04, c4= np.polyfit(T4, B_ofw, 1)
m05, c5= np.polyfit(T5, B_ofw, 1)
m06, c6= np.polyfit(T6, B_ofw, 1)
m07, c7= np.polyfit(T7, B_ofw, 1)
m08, c8= np.polyfit(T8, B_ofw, 1)
m09, c9= np.polyfit(T9, B_ofw, 1)


#Calculates Activation Energy from slopes
m_kasarr=np.asarray([m1,m2,m3,m4,m5,m6,m7,m8,m9])
m_ofwarr=np.asarray([m01,m02,m03,m04,m05,m06,m07,m08,m09])
Ea_KAS=(-1*m_kasarr)*(8.314/1000)
Ea_OFW=(-1*m_ofwarr)*(8.314/(1.052*1000))

#creates column of full alpha data (x), dropping duplicate values
#also creates evenly spaced time array of the same size as x
x1=TGA_2c['alpha 2c'].drop_duplicates(keep='first')
y1=np.linspace(100,29128,1002)
x2=TGA_4c['alpha 4c'].drop_duplicates(keep='first')
y2=np.linspace(100,14533,1002)
x3=TGA_6c['alpha 6c'].drop_duplicates(keep='first')
y3=np.linspace(100,9689,1002)
x4=TGA_8c['alpha 8c'].drop_duplicates(keep='first')
y4=np.linspace(100,7267,1002)

#time derivative of alpha and natural log of these arrays
dadt2=diff(x1)/diff(y1)
dadt4=diff(x2)/diff(y2)
dadt6=diff(x3)/diff(y3)
dadt8=diff(x4)/diff(y4)
lnda2=np.log(dadt2)[0:9]
lnda4=np.log(dadt4)[0:9]
lnda6=np.log(dadt6)[0:9]
lnda8=np.log(dadt8)[0:9]

#inverse temperature data (Kelvin) for each heating rate experiment
T2f=(Alpha1['Temp (C) 2c']+273)**-1
T4f=(Alpha2['Temp (C) 4c']+273)**-1
T6f=(Alpha3['Temp (C) 6c']+273)**-1
T8f=(Alpha4['Temp (C) 8c']+273)**-1

#creates array of ln(dadt) and inverse temperature
Arr2=pd.concat([pd.DataFrame(lnda2),pd.DataFrame(T2f)], axis=1, ignore_index=True)
Arr4=pd.concat([pd.DataFrame(lnda4),pd.DataFrame(T4f)], axis=1, ignore_index=True)
Arr6=pd.concat([pd.DataFrame(lnda6),pd.DataFrame(T6f)], axis=1, ignore_index=True)
Arr8=pd.concat([pd.DataFrame(lnda8),pd.DataFrame(T8f)], axis=1, ignore_index=True)

#resturctures data into each alpha value
r11=np.asarray(pd.concat([Arr2[:1],Arr4[:1],Arr6[:1],Arr8[:1]]))
r21=np.asarray(pd.concat([Arr2[1:2],Arr4[1:2],Arr6[1:2],Arr8[1:2]]))
r31=np.asarray(pd.concat([Arr2[2:3],Arr4[2:3],Arr6[2:3],Arr8[2:3]]))
r41=np.asarray(pd.concat([Arr2[3:4],Arr4[3:4],Arr6[3:4],Arr8[3:4]]))
r51=np.asarray(pd.concat([Arr2[4:5],Arr4[4:5],Arr6[4:5],Arr8[4:5]]))
r61=np.asarray(pd.concat([Arr2[5:6],Arr4[5:6],Arr6[5:6],Arr8[5:6]]))
r71=np.asarray(pd.concat([Arr2[6:7],Arr4[6:7],Arr6[6:7],Arr8[6:7]]))
r81=np.asarray(pd.concat([Arr2[7:8],Arr4[7:8],Arr6[7:8],Arr8[7:8]]))
r91=np.asarray(pd.concat([Arr2[8:9],Arr4[8:9],Arr6[8:9],Arr8[8:9]]))

#linear model for curve fitting
def objective(x,a,b,c):
    return a*x+b

#uses scipy curve fitting to fins regression of data
popt1, _ =curve_fit(objective,r11[:,1],r11[:,0])
m1f,c1f,k1f=popt1

popt2, _ =curve_fit(objective,r21[:,1],r21[:,0])
m2f,c2f,k2f=popt2

popt3, _ =curve_fit(objective,r31[:,1],r31[:,0])
m3f,c3f,k3f=popt3

popt4, _ =curve_fit(objective,r41[:,1],r41[:,0])
m4f,c4f,k4f=popt4

popt5, _ =curve_fit(objective,r51[:,1],r51[:,0])
m5f,c5f,k1f=popt5

popt6, _ =curve_fit(objective,r61[:,1],r61[:,0])
m6f,c6f,k6f=popt6

popt7, _ =curve_fit(objective,r71[:,1],r71[:,0])
m7f,c7f,k7f=popt7

popt8, _ =curve_fit(objective,r81[:,1],r81[:,0])
m8f,c8f,k8f=popt8

popt9, _ =curve_fit(objective,r91[:,1],r91[:,0])
m9f,c9f,k9f=popt9

#calculates Friedman activation energy from slopes
mArr_friedmann=np.asarray([m1f,m2f,m3f,m4f,m5f,m6f,m7f,m8f,m9f])
Eafriedmann=(-1*mArr_friedmann)*(8.314/1000)

#plots all activation energy values wrt alpha
plt.plot(alpha1, Ea_KAS, '.', color='m',linestyle= '-.',mfc='white',mec='black', label="KAS method")
plt.ylabel('Ea (kJ)')
plt.xlabel('Alpha')
plt.plot(alpha1, Ea_OFW,  color='b',marker='v',linestyle= '-',mfc='white',mec='black', label="OFW method")
plt.plot(alpha1, Eafriedmann,  color='g',marker='D', linestyle='--',mfc='white',mec='black', label="Friedman method")
plt.title('Ea calcined 5% H2 ')
plt.legend()
plt.show()

#puts all Ea into single frame and saves as csv to working directory
Ea_KAS=pd.DataFrame(Ea_KAS)
Ea_OFW=pd.DataFrame(Ea_OFW)
Eafriedmann=pd.DataFrame(Eafriedmann)
Ea_arr=pd.concat([Ea_KAS, Ea_OFW,Eafriedmann], axis=1).set_axis(['KAS','OFW','Friedman'],axis=1)

Ea_arr.to_csv('#Ea.csv')