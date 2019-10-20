''' 
In this code I am going to explore historical weather data from Oxford
'''
sdfs

# Importing the relevent libraries and ensuring all plots are closed
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.close('all')


# Reading the data in 
df = pd.read_csv('weather_data.csv')

# First step of data cleaning, removing unwanted symbols from columns using 
# left strip, right strip applied to all rows in the frame using map and lambda.
df['tmin (degC)'] = df['tmin (degC)'].map(lambda x: x.lstrip(' *'))
df['air frost (days)'] = df['air frost (days)'].map(lambda x: x.lstrip(' -*'))
df['rain (mm)'] = df['rain (mm)'].map(lambda x: x.lstrip('- *'))
df['sun (hours)'] = df['sun (hours)'].map(lambda x: x.lstrip('- *'))
df['sun (hours)'] = df['sun (hours)'].map(lambda x: x.rstrip('- * Provisional'))

# Now to convert the rows from Objects to floats to enable us to be able to manipulate the data
# errors set to coerce ensures that missing/corrupted data is ignored
df['tmin (degC)'] = pd.to_numeric(df['tmin (degC)'],errors='coerce')
df['air frost (days)'] = pd.to_numeric(df['air frost (days)'],errors='coerce')
df['rain (mm)'] = pd.to_numeric(df['rain (mm)'] ,errors='coerce')
df['sun (hours)'] = pd.to_numeric(df['sun (hours)'],errors='coerce')

# If I wanted to replace all the numerical months with letters ...
# df['mm'].replace(1,'Jan')

# Lets start by exploring some of the data 
# First making a scatter plot of monthly temperature distribution
fig,axes = plt.subplots(1,2,sharey=True)
axes[0].scatter(df['mm'],df['tmin (degC)'])
axes[1].scatter(df['mm'],df['tmax (degC)'], color = 'red')
axes[0].set_ylabel('Temperature (degC)')
axes[0].set_title('Tmin')
axes[1].set_title('Tmax')
axes[0].set_xlabel('Month')
axes[1].set_xlabel('Month')

# Setting the x axis to replace the numbers with letters for months
month_names = ['J','F','M','A','M','J','J','A','S','O','N','D']
axes[0].set_xticks(np.arange(1,13))
axes[0].set_xticklabels(month_names)
axes[1].set_xticks(np.arange(1,13))
axes[1].set_xticklabels(month_names)

# Would be cool to add a running mean through the middle of them 

# TIDY THIS UP

df1 = df[df['yyyy'] < 1900]
df2 = df[(df['yyyy']>1900) & (df['yyyy']<1950)]
df3 = df[(df['yyyy']>1950) & (df['yyyy']<2000)]
df4 = df[(df['yyyy']>2000) & (df['yyyy']<2018)]

df_mm_mean_1900 = df1.groupby('mm',as_index=False,).mean()
df_mm_mean_2018 = df4.groupby('mm',as_index=False,).mean()
df_mm_mean = df.groupby('mm',as_index=False,).mean()

axes[0].plot(
	df_mm_mean['mm'],
	df_mm_mean['tmin (degC)'],
	color='blue',
	label = 'mean')

axes[0].plot(
	df_mm_mean_1900['mm'],
	df_mm_mean_1900['tmin (degC)'],
	color='orange',
	label = '1853-1900')

axes[0].plot(
	df_mm_mean_2018['mm'],
	df_mm_mean_2018['tmin (degC)'],
	color='red',
	label = '2000-2018')

axes[0].legend()

axes[1].plot(
	df_mm_mean['mm'],
	df_mm_mean['tmax (degC)'],
	color = 'red')

axes[1].legend()

# Creating a pivot table to allow for  visualisations via heatmaps
plt.figure()
df_piv1 = df.pivot_table(
	index = 'mm', 
	columns = 'yyyy', 
	values = 'tmax (degC)')
sns.heatmap(df_piv1,
	cmap = 'coolwarm')

plt.figure()

df_yyyy_mean = df.groupby('yyyy').mean()
df_yyyy_mean.drop(2019, inplace=True)
df_yyyy_mean_pivot = df_yyyy_mean.pivot_table(
	index='mm',
	columns = 'yyyy',
	values='tmax (degC)')
sns.heatmap(df_yyyy_mean_pivot,
	cmap = 'coolwarm')
# need to get rid of y axis here

plt.show()