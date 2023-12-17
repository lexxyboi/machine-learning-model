# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('bitcoin_price.csv')

# Split the dataset for input and output
X = df[['Open', 'High', 'Low']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=72)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

with open('regression.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
