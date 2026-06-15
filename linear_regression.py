import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Datasets
wv_df = pd.read_csv(r'D:\Hustle\Used-Car-Price-Prediction\Datasets\vw.csv')
audi_df = pd.read_csv(r'D:\Hustle\Used-Car-Price-Prediction\Datasets\audi.csv')

# Adding brand column to each datasets

wv_df['brand'] = 'Volkswagen'
audi_df['brand'] = 'Audi'

# Combining both Datasets into one dataset

df = pd.concat([wv_df, audi_df], ignore_index=True)

# Model Training

# 1. Define X (mileage) and y (price)
x = df[['mileage']]
y = df[['price']]

#2. Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)  

# 3. Fit LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# 4. Calculate MAE
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(mae.__round__())