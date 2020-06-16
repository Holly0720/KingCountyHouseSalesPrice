#!/usr/bin/env python
# coding: utf-8

# # HouseSales in KingCounty, USA
# 
# data sources:https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# tutorial sources: Coursera - Data Analysis with Python
# 
# data descriptions: https://rstudio-pubs-static.s3.amazonaws.com/155304_cc51f448116744069664b35e7762999f.html
# 
# id - Unique ID for each home sold
# 
# date - Date of the home sale
# 
# price - Price of each home sold
# 
# bedrooms - Number of bedrooms
# 
# bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# 
# sqft_living - Square footage of the apartments interior living space
# 
# sqft_lot - Square footage of the land space
# 
# floors - Number of floors
# 
# waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# 
# view - An index from 0 to 4 of how good the view of the property was
# 
# condition - An index from 1 to 5 on the condition of the apartment,
# 
# grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# 
# sqft_above - The square footage of the interior housing space that is above ground level
# 
# sqft_basement - The square footage of the interior housing space that is below ground level
# 
# yr_built - The year the house was initially built
# 
# yr_renovated - The year of the house’s last renovation
# 
# zipcode - What zipcode area the house is in
# 
# lat - Lattitude
# 
# long - Longitude
# 
# sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# 
# sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score,cross_val_predict


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


housesales = pd.read_csv('/kcsales/kc_house_data.csv')


# In[4]:


housesales.head()


# In[5]:


housesales.info()


# In[6]:


# delete useless columns
housesales.drop(['id'],axis=1,inplace=True)


# In[7]:


# change datatypes
housesales['date']=pd.to_datetime(housesales['date']).dt.strftime('%Y-%m')
housesales['zipcode']=housesales['zipcode'].astype(object)


# In[8]:


housesales.describe()


# In[9]:


# date
fig = plt.figure(figsize=(12,4)) 
sns.lineplot(x='date',y='price',data=housesales)
plt.xticks(rotation=-30)


# In[10]:


# bedrooms
print(housesales['bedrooms'].value_counts())
sns.countplot(housesales['bedrooms'])


# from the graph of bedrooms, there is one value of 33 bedrooms, which is extreme and can be deleted. 75% of houses have no more than 4 bedrooms.

# In[11]:


housesales.drop(housesales[housesales['bedrooms']>30].index,axis=0,inplace=True)


# In[12]:


# bathrooms
housesales['bathrooms'].value_counts()
sns.distplot(housesales['bathrooms'],bins=10,kde=False)


# More than 75% of houses have no more than 3 bathrooms.

# In[13]:


# sqft_living
sns.regplot(x='sqft_living',y='price',data=housesales)


# We can see, the sqft_living and price has a positive relationship. We can calculate the correlation and p_value for them. There is a value, whose sqft_living more than 12000, but price is lower. Maybe we can delete it.

# In[14]:


# delete the extrme value and calculate the correlation between sqft_living and price
housesales.drop(housesales['sqft_living'].argmax(),axis=0,inplace=True)
person_coef,p_value = stats.pearsonr(housesales['sqft_living'],housesales['price'])
print('the correlation between sqft_living and price is ',person_coef,'and the p_value is ',p_value)


# In[15]:


# sqft_lot
sns.regplot(x='sqft_lot',y='price',data=housesales)
person_coef,p_value = stats.pearsonr(housesales['sqft_lot'],housesales['price'])
print('the correlation between sqft_lot and price is ',person_coef,'and the p_value is ',p_value)


# It seems like that price is the sqft_lot's power function result, y=x^a(a<0). 
# They are not significantly correlated.

# In[16]:


# floors
sns.barplot(x='floors',y='price',data=housesales,estimator=np.median)


# The floors with 2.5 have higher median of prices than others, which are mostly similar.

# In[17]:


# waterfront
sns.boxplot(x='waterfront',y='price',data=housesales)


# The distribution of prices between different waterfront, 0 or 1, are significantly different, which the 1s have higher prices. Maybe waterfront is a good predicter.

# In[18]:


# view
sns.boxplot(x='view',y='price',data=housesales)


# The houses with 1 and 2 in view field have similar prices, higher than 0. About 50% of houses with 4 have higher prices than 0,1,2,3. In general, houses with better views have higher prices.

# In[19]:


# condition
sns.boxplot(x='condition',y='price',data=housesales)


# The distribution of prices in different conditions are not significantly different, so maybe condition is not a good variable for the prediction of price.

# In[20]:


# grade
sns.barplot(x='grade',y='price',data=housesales,estimator=np.median)


# From the graph, we can see, as the median of grade becomes higher, the price also becomes higher. Maybe it is a good feature.

# In[21]:


# sqft_above
sns.regplot(x='sqft_above',y='price',data=housesales)
person_coef,p_value = stats.pearsonr(housesales['sqft_above'],housesales['price'])
print('the correlation between sqft_above and price is ',person_coef,'and the p_value is ',p_value)


# The correlation is about 0.60, and p_value is 0.0. We can say, the sqft_above and price have significantly positive relationship, which is also shown in the graph.

# In[22]:


# sqft_basement
sns.regplot(x='sqft_basement',y='price',data=housesales)
person_coef,p_value = stats.pearsonr(housesales['sqft_basement'],housesales['price'])
print('the correlation between sqft_basement and price is ',person_coef,'and the p_value is ',p_value)


# The correlation between sqft_basement and price is less than 0.5, but significantly positive. Maybe we can cconsider it in the modeling.

# In[23]:


# yr_built
fig = plt.figure(figsize=(20,4)) 
ax = sns.scatterplot(x='yr_built',y='price',data=housesales)
ax.set_xticks(np.arange(1900,2030,10))
plt.xticks(rotation=-30)


# From the scatterplot between yr_built and price, we see: there are 2 extrme values in 1910 and 1940, whose prices are much higher than others, more than 7000000. We can find them out and check the reason.
# Maybe the reason is the grades of these 2 data are better than others in the same year.

# In[24]:


housesales[housesales['price']>7000000]


# In[25]:


price1910 = housesales[(housesales['yr_built']==1910) & (housesales['price']<700000)]
price1910.describe()


# In[26]:


price1940 = housesales[(housesales['yr_built']==1940) & (housesales['price']<700000)]
price1940.describe()


# In[27]:


# yr_renovated
print('The persentage of yr_renovated with zero is ',(housesales['yr_renovated']==0).sum()/housesales['yr_renovated'].count())


# More than 95% of houses have 0 year_renovated, which means none. (since we don't know the exact data description, none is just the guess.)

# In[28]:


# lat & long
fig = plt.figure(figsize=(30,20)) 
plt.subplot(2,2,1)
sns.distplot(housesales['lat'],kde=False)
plt.subplot(2,2,2)
sns.distplot(housesales['long'],kde=False)
plt.subplot(2,2,3)
sns.barplot(housesales['price'],pd.cut(housesales['lat'],bins=6,labels=None,include_lowest=True))
#plt.xticks(rotation=-30)
plt.subplot(2,2,4)
sns.barplot(housesales['price'],pd.cut(housesales['long'],bins=6,labels=None,include_lowest=True))
#plt.xticks(rotation=-30)


# We can see that: the houses with lattitude between 47.57 to 47.674 have the highest average of prices, and with longitube between -121.516 to -121.315 have the lowest average of prices

# In[29]:


# add interesting houses in a map
#latlong = housesales[['lat','long']]
#mapit = folium.Map(location=[47.560053, -122.257])

#for i in range(len(latlong)): 
#    folium.Marker(location=[latlong.loc[i,'lat'],latlong.loc[i,'long']], fill_color='#43d9de', radius=8).add_to(mapit)

#folium.Marker(location=[47.5112,-122.257], fill_color='#43d9de', radius=8).add_to(mapit)
#mapit


# In[30]:


# sqft_living15
sns.regplot(x='sqft_living15',y='price',data=housesales) #lmplot
person_coef,p_value = stats.pearsonr(housesales['sqft_living15'],housesales['price'])
print('the correlation between sqft_living15 and price is ',person_coef,'and the p_value is ',p_value)


# The sqft_living15 may be a good predictor, since it has significantly positive correlation with price.

# In[31]:


# sqft_lot15
sns.regplot(x='sqft_lot15',y='price',data=housesales)
person_coef,p_value = stats.pearsonr(housesales['sqft_lot15'],housesales['price'])
print('the correlation between sqft_lot15 and price is ',person_coef,'and the p_value is ',p_value)


# en...It is not a good variable.

# In[32]:


print(housesales.corr()['price'])
sns.heatmap(housesales.corr(),cmap="YlGnBu")


# From the correlation among features and prices, and the graphs shown before, the features which are below may be the good predictors of price. So, we choose these to built the model and predict:
# - bedrooms
# - bathrooms
# - sqft_living
# - floors
# - waterfront
# - view
# - grade
# - sqft_above
# - sqft_basement
# - lat
# - sqft_living15

# In[52]:


# data modelling
# from sklearn.preprocessing import StandardScaler
X = housesales[['bedrooms','bathrooms','sqft_living','floors','waterfront','view','grade','sqft_above','sqft_basement','lat','sqft_living15']]
y = housesales[['price']]

# Scaling the data
scale = StandardScaler()
X_scale = StandardScaler.fit_transform(scale,X)
X_scale_train,X_scale_test,y_train,y_test = train_test_split(X_scale,y,random_state=0)

# Linear Regression
lm = LinearRegression()
lm.fit(X_scale_train,y_train)
y_estimated = lm.predict(X_scale)

# print the coefficient and intercept
label = ['bedrooms','bathrooms','sqft_living','floors','waterfront','view','grade','sqft_above','sqft_basement','lat','sqft_living1']
coefdf = pd.DataFrame()
coefdf['label'] = label
coefdf['coefficient'] = lm.coef_[0]
print(coefdf)
print('\nThe intercept of model is ',lm.intercept_[0])

# R_squared
print('The R-squared of test data is ',lm.score(X_scale_test,y_test))

# MSE
from sklearn.metrics import mean_squared_error
print('The MSE is ',mean_squared_error(y,y_estimated))

# plot the residual plot
plt.figure(figsize=(10,10))
ax1 = sns.distplot(housesales['price'],hist=False,color='r',label='actual price')
sns.distplot(y_estimated,hist=False,color='b',label='predicted price',ax=ax1)


# We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, the distribution of price are different between 0 to 100000, and the R-squared is about 0.64 and MSE is too large.
# 
# Therefore, there is definitely some room for improvement.
# 
# The next step, we can try fitting a polynomial model to the data instead. 

# In[54]:


# Polynomial Linear Regression

#from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_scale_poly = PolynomialFeatures.fit_transform(poly,X_scale)

X_scale_poly_train,X_scale_poly_test,y_train,y_test = train_test_split(X_scale_poly,y,random_state=0)

lm.fit(X_scale_poly_train,y_train)
print('The R-squared of test data is ',lm.score(X_scale_poly_test,y_test))

# plot the residual plot
plt.figure(figsize=(10,10))
ax2 = sns.distplot(housesales['price'],hist=False,color='r',label='actual price')
sns.distplot(lm.predict(X_scale_poly),hist=False,color='b',label='predicted price(poly_linear)',ax=ax2)


# We can also use the Pipline to create the process of scaling, polynomial transforming and modeling.

# In[56]:


# Creat Pipline for scaling the data to perform a polynomial transform
#from sklearn.pipeline import Pipeline

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2,include_bias=False)),('lr',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X_train,y_train)
print('The R-squared of test data is ',pipe.score(X_test,y_test))
print(pipe.named_steps['lr'].coef_)


# We can see that the R-squared of the polynomial linear regression is better than linear regression. But the distribution of price in (0-1000000) is still different - the number of fitted values are more than the actual values in this period.
# 
# We also see there are some so large coefficients...
# 
# Let's try the Ridge regression with standard scale and polynomial linear regression.

# In[57]:


# Redge regression with standard scale and polynomial linear regression - prevents overfitting

# with pipeline
Input_ridge = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2,include_bias=False)),('ridge',Ridge(alpha=0.1))]
pipe_ridge = Pipeline(Input_ridge)
pipe_ridge.fit(X_train,y_train)
print('The R-squared of test data is ',pipe_ridge.score(X_test,y_test))
print(pipe_ridge.named_steps['ridge'].coef_)

# plot the residual plot
plt.figure(figsize=(10,10))
ax3 = sns.distplot(housesales['price'],hist=False,color='r',label='actual price')
sns.distplot(pipe_ridge.predict(X),hist=False,color='b',label='predicted price(ridge)',ax=ax3)


# After using the ridge regression instead of linear regression, we can see the coefficient are much smaller than before, and the R-squared is similar.

# In[62]:


# Grid Search to find the best alpha of Ridge regression
#from sklearn.model_selection import GridSearchCV
#parameter = [{'alpha':[0.001,0.01,0.1,1]}]

#ridge = Ridge()
#grid = GridSearchCV(ridge,parameter,cv=5)
#grid.fit(X_scale_poly,y)

#print('The best alpha value is ',grid.best_estimator_)
#print('The R-squared for test data and different alpha is ',grid.cv_results_['mean_test_score'])


# In[72]:


# Model evaluation
from sklearn.model_selection import cross_val_score,cross_val_predict

# linear Regression
Rcross_lm = cross_val_score(lm,X_scale,y,cv=5)
print("The mean of the folds(lm) are", Rcross_lm.mean(), "and the standard deviation is" , Rcross_lm.std())

# Polynomial Regression
Rcross_poly = cross_val_score(lm,X_scale_poly,y,cv=5)
print("The mean of the folds(poly_lm) are", Rcross_poly.mean(), "and the standard deviation is" , Rcross_poly.std())

# Ridge+Polynomial
ridge = Ridge(alpha=0.1)
Rcross_ridge = cross_val_score(ridge,X_scale_poly,y,cv=5)
print("The mean of the folds(ridge_poly) are", Rcross_ridge.mean(), "and the standard deviation is" , Rcross_ridge.std())


# From the results of model evaluation, the ridge regression with standardscale and polynomial is better than other models.
# 
# So, we can choose it as the final model to predict the house sales price.

# In[ ]:




