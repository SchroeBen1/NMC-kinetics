import pandas as pd
import matplotlib.pyplot as plt

#%%
#define file path where data is located
TGA_path='#filepath'
#define column names for dataframes
headerlist_1=["Time (min)", "Temp (C) 2c", "Heat Flow (W/g)", "Weight (%)", "Heat Flow (mW)", "Weight (mg)", "alpha 2c"]
headerlist_2=["Time (min)", "Temp (C) 4c", "Heat Flow (W/g)", "Weight (%)", "Heat Flow (mW)", "Weight (mg)", "alpha 5c"]
headerlist_3=["Time (min)", "Temp (C) 6c", "Heat Flow (W/g)", "Weight (%)", "Heat Flow (mW)", "Weight (mg)", "alpha 10c"]
headerlist_4=["Time (min)", "Temp (C) 8c", "Heat Flow (W/g)", "Weight (%)", "Heat Flow (mW)", "Weight (mg)", "alpha 15c"]
#use read_csv to import data in .csv format for each heating rate
#first 1000 columns are dropped as they are noise from TGA and rounds alpha data to 3 decimals
TGA_2c=pd.read_csv(TGA_path+'#heating data 1',names=headerlist_1)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Heat Flow (mW)', 'Weight (mg)','alpha 2c'], axis='columns')
TGA_2c=TGA_2c.drop(TGA_2c.index[0:1000]).astype(float).round(3)
TGA_4c=pd.read_csv(TGA_path+'#heating data 2',names=headerlist_2)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Heat Flow (mW)', 'Weight (mg)', 'alpha 5c'], axis='columns')
TGA_4c=TGA_4c.drop(TGA_4c.index[0:1000]).astype(float).round(3)
TGA_6c=pd.read_csv(TGA_path+'#heating data 3',names=headerlist_3)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Heat Flow (mW)', 'Weight (mg)','alpha 10c'], axis='columns')
TGA_6c=TGA_6c.drop(TGA_6c.index[0:1000]).astype(float).round(3)
TGA_8c=pd.read_csv(TGA_path+'#heating data 4',names=headerlist_4)\
.drop(['Time (min)', 'Heat Flow (W/g)', 'Heat Flow (mW)', 'Weight (mg)', 'alpha 15c'], axis='columns')
TGA_8c=TGA_8c.drop(TGA_8c.index[0:1000]).astype(float).round(3)

#%%
#combines data into frame and create x,y pairs of weight and time to plot
#then plots it
FullFrame=pd.concat([TGA_2c,TGA_4c,TGA_6c,TGA_8c], axis=1)
x1=TGA_2c['Temp (C) 2c']
y1=TGA_2c['Weight (%)']
plt.plot(x1,y1, '--', label='2c/min')
x2=TGA_4c['Temp (C) 4c']
y2=TGA_4c['Weight (%)']
plt.plot(x2,y2 , label='4c/min')
x3=TGA_6c['Temp (C) 6c']
y3=TGA_6c['Weight (%)']
plt.plot(x3,y3, '-.', label='6c/min')
x4=TGA_8c['Temp (C) 8c']
y4=TGA_8c['Weight (%)']
plt.plot(x4,y4, ':', label='8c/min')
plt.legend()
plt.xlabel('Temp (C)')
plt.ylabel('weight (%)')
plt.title('5% Calcined weight change ')
plt.show()


