# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
data = pd.read_stata('karlan_list_2007.dta')
data.describe()
# print(np.sum(data['treatment']==1))
# print(np.sum(data['control']==0))

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
data.isna().sum()

#
#
#
#

data= data.dropna()
data.shape
```
#

data.head()
```
#
#
m1= data[data['treatment']==1]['statecnt'].mean()
m2= data[data['treatment']==0]['statecnt'].mean()
n1= data[data['treatment']==1]['statecnt'].count()
n2= data[data['treatment']==0]['statecnt'].count()
s1= data[data['treatment']==1]['statecnt'].std()
s2= data[data['treatment']==0]['statecnt'].std()
se= np.sqrt((s1**2)/n1+(s2**2)/n2)
t= (m1-m2)/se
import scipy.stats as stats
stats.ttest_ind(a=data[data['treatment']==1]['statecnt'],b=data[data['treatment']==0]['statecnt'],equal_var=False)
```
#
print(f'difference in means: {m1-m2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')
#
#
#
#
#
from sklearn.linear_model import LinearRegression
import statsmodels.api as smf
X = smf.add_constant(data['treatment'])

x= np.array(data['treatment']).reshape(-1,1)
y= data['statecnt']
model = smf.OLS(y,X)
results = model.fit()
print(results.summary())




#
#
#
resids = np.array(y-model.predict(x))
model2= LinearRegression()
model2.fit(resids.reshape(-1,1),x.flatten())
model2.coef_

#
#
#
#
#
m1= data[data['treatment']==1]['mrm2'].mean()
m2= data[data['treatment']==0]['mrm2'].mean()
n1= data[data['treatment']==1]['mrm2'].count()
n2= data[data['treatment']==0]['mrm2'].count()
s1= data[data['treatment']==1]['mrm2'].std()
s2= data[data['treatment']==0]['mrm2'].std()
se= np.sqrt((s1**2)/n1+(s2**2)/n2)
t= (m1-m2)/se
import scipy.stats as stats
stats.ttest_ind(a=data[data['treatment']==1]['mrm2'],b=data[data['treatment']==0]['mrm2'],equal_var=False)
```
#
print(f'difference in means: {m1-m2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')
#
#
#
#
from sklearn.linear_model import LinearRegression
x= np.array(data['treatment']).reshape(-1,1)
y= data['mrm2']
model = LinearRegression()
model.fit(x,y)
print(f'regression coefficient: {model.coef_[0]:.3f}')

#
#
#
resids = np.array(y-model.predict(x))
model2= LinearRegression()
model2.fit(resids.reshape(-1,1),x.flatten())
model2.coef_

```
#
#
#
#
#
#
#
#
#
#
import seaborn as sns 
sns.barplot(data = data, x= 'treatment', y = 'gave', estimator= 'mean')


#
#
#
p1= data[data['treatment']==1]['gave'].mean()
p2= data[data['treatment']==0]['gave'].mean()
n1= data[data['treatment']==1]['gave'].count()
n2= data[data['treatment']==0]['gave'].count()
se= np.sqrt((p1*(1-p1)/n1+(p2*(1-p2))/n2))
t= (p1-p2)/se
import scipy.stats as stats
stats.ttest_ind(a=data[data['treatment']==1]['gave'],b=data[data['treatment']==0]['gave'],equal_var=False)

```
#


print(f'difference in proportions: {p1-p2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')



#
#
#
#

import statsmodels.api as smf
x = smf.add_constant(data['treatment'])
model3 = smf.Probit(data['gave'], x)
result = model3.fit()
result.summary()





```
#
#
#
#
#
#

p1= data[data['ratio']==1]['gave'].mean()
p2= data[data['ratio2']==1]['gave'].mean()
n1= data[data['ratio']==1]['gave'].count()
n2= data[data['ratio2']==1]['gave'].count()
se= np.sqrt((p1*(1-p1)/n1+(p2*(1-p2))/n2))
t= (p1-p2)/se

print(f'difference in proportions: {p1-p2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')



#
#
#
p1= data[data['ratio']==1]['gave'].mean()
p2= data[data['ratio3']==1]['gave'].mean()
n1= data[data['ratio']==1]['gave'].count()
n2= data[data['ratio3']==1]['gave'].count()
se= np.sqrt((p1*(1-p1)/n1+(p2*(1-p2))/n2))
t= (p1-p2)/se

print(f'difference in proportions: {p1-p2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')



#
#
#
p1= data[data['ratio2']==1]['gave'].mean()
p2= data[data['ratio3']==1]['gave'].mean()
n1= data[data['ratio2']==1]['gave'].count()
n2= data[data['ratio3']==1]['gave'].count()
se= np.sqrt((p1*(1-p1)/n1+(p2*(1-p2))/n2))
t= (p1-p2)/se

print(f'difference in proportions: {p1-p2:.3f}')
print(f't-statistic: {t:.3f}')
print(f'p-value: {2*(1-stats.t.cdf(np.abs(t),n1+n2)):.3f}')



#
#
#
#
#
#
#data['ratio1']= data['ratio'].astype(float)-data['ratio2']-data['ratio3']
#data2= data[data['treatment']==1]
data['ratio1']= np.where((data['ratio']!='Control') & (data['ratio2']==0) & (data['ratio3']==0),1,0)
X = smf.add_constant(data[['ratio1','ratio2','ratio3']])
Y = data['gave']
model4 = smf.OLS(Y,X)
results = model4.fit()
results.summary()



#
#
#
#
#
#

print(data[data['ratio2']==1]['gave'].mean()-data[data['ratio1']==1]['gave'].mean())
print(data[data['ratio3']==1]['gave'].mean()-data[data['ratio2']==1]['gave'].mean())
print(results.params['ratio2']-results.params['ratio1'])
print(results.params['ratio3']-results.params['ratio2'])



#
#
#
#
#
#
#
#
#
#
#
X=smf.add_constant(data['treatment'])
y= data['amount']
model = smf.OLS(y,X)
results = model.fit()
results.summary()

```
#
#
#
#
data3 = data[data['gave']==1]
X=smf.add_constant(data3['treatment'])
y= data3['amount']
model = smf.OLS(y,X)
results = model.fit()
results.summary()


```
#
#
#
#
#
import matplotlib.pyplot as plt 
treat = data3[data3['treatment']==1]
control = data3[data3['treatment']==0]
sns.histplot(treat, x = 'amount')
plt.axvline(x = treat['amount'].mean(), color = 'r')

#
#
#
sns.histplot(control, x = 'amount')
plt.axvline(x = control['amount'].mean(), color = 'r')

```
#
#
#
#
#
#
#
#
#
#
#
#
control = stats.bernoulli.rvs(0.018, size = 10000)
treat = stats.bernoulli.rvs(0.022, size = 10000)
diff = treat - control 
cumm_mean = np.cumsum(diff)/np.arange(1, 10001)
sns.lineplot(cumm_mean, color = 'r')
plt.axhline(y = 0.004, linestyle = '--' )
#
#
#
#
#
#
#
#

avg50 = []
for i in range(1000):
    control = stats.bernoulli.rvs(0.018, size = 50)
    treat = stats.bernoulli.rvs(0.022, size = 50)
    diff = treat - control 
    avg50.append(np.mean(diff))

cumm_mean = np.cumsum(avg50)/np.arange(1,1001)
sns.lineplot(cumm_mean)
plt.axhline(y = 0.004, linestyle = '--' )
plt.hist(x = avg50, orientation = 'horizontal')


#
#
#

avg200 = []
for i in range(1000):
    control = stats.bernoulli.rvs(0.018, size = 200)
    treat = stats.bernoulli.rvs(0.022, size = 200)
    diff = treat - control 
    avg200.append(np.mean(diff))

cumm_mean = np.cumsum(avg200)/np.arange(1,1001)
sns.lineplot(cumm_mean)
plt.axhline(y = 0.004, linestyle = '--' )
plt.hist(x = avg200, orientation = 'horizontal')


#
#
#
#

avg500 = []
for i in range(1000):
    control = stats.bernoulli.rvs(0.018, size = 500)
    treat = stats.bernoulli.rvs(0.022, size = 500)
    diff = treat - control 
    avg500.append(np.mean(diff))

cumm_mean = np.cumsum(avg500)/np.arange(1,1001)
sns.lineplot(cumm_mean)
plt.axhline(y = 0.004, linestyle = '--' )
plt.hist(x = avg500, orientation = 'horizontal')


```
#

avg1000 = []
for i in range(1000):
    control = stats.bernoulli.rvs(0.018, size = 1000)
    treat = stats.bernoulli.rvs(0.022, size = 1000)
    diff = treat - control 
    avg1000.append(np.mean(diff))

cumm_mean = np.cumsum(avg500)/np.arange(1,1001)
sns.lineplot(cumm_mean)
plt.axhline(y = 0.004, linestyle = '--' )
plt.hist(x = avg1000, orientation = 'horizontal')


#
#
#
#
