import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv('insurance_data2.csv')
print(df)
#fetching value from the file and store in x and y
x= df.iloc[:,0:1].values
y=df.iloc[:,1].values
print(x)
print(y)

#to visualize the data
plt.title("insurance data")
plt.xlabel("age")
plt.ylabel("insured or not")
plt.scatter(x,y)
plt.show()

#calculation for that regression and that best fit line
n=len(x)
m_x= np.mean(x)
m_y=np.mean(y)

SS_xy= np.sum(x*y)-n *m_x*m_y
SS_xx= np.sum(x*x)-n *m_x*m_x

m=SS_xy/SS_xx
print("value of m ",m)

c= m_y - m*m_x
print("value of c= ",c)

print("\nModel: y= ", m,"*x+",c)  #regression formula
#applied the formula tbo the past data to check the reliability
y_predicted= m*x+c
print(y_predicted)

plt.title("insurance data")
plt.xlabel("age")
plt.ylabel("insured or not")
plt.scatter(x,y)
plt.plot(x,y_predicted,"ro-")
plt.show()

import math
def get_sigmoid(z):  #below in the get sigmoid part i have put the value of z which is predicted_y
 return 1/(1+ np.power(math.e,-z))
plt.title("insurance data")
plt.xlabel("age")
plt.ylabel("insured or not")
plt.scatter(x,y,color="b", s=150)
plt.plot(x, get_sigmoid(y_predicted),"ro-")
plt.show()