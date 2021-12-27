from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv("escape_velocity.csv")

velocity_list = df["Velocity"].tolist()
escaped_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=escaped_list)
fig.show()


velocity_array = np.array(velocity_list)
escaped_array = np.array(escaped_list)

#Slope and intercept using pre-built function of Numpy
m, c = np.polyfit(velocity_array, escaped_array, 1)

y = []
for x in velocity_array:
  y_value = m*x + c
  y.append(y_value)

#plotting the graph
fig = px.scatter(x=velocity_array, y=escaped_array)
fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(velocity_array), x1= max(velocity_array)
    )
])
fig.show()

X = np.reshape(velocity_list, (len(velocity_list), 1))
Y = np.reshape(escaped_list, (len(escaped_list), 1))

lr = LogisticRegression()
lr.fit(X, Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='black', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

#Using the line formula 
X_test = np.linspace(0, 5000, 10000)
melting_chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, melting_chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

#do hit and trial by changing the vlaue of X_test here.
plt.axvline(x=X_test[6843], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(3400, 3450)
plt.show()
print(X_test[6843])