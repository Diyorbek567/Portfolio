import pandas as pd

df = pd.read_csv('train.csv')
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 150)
print(df.head(2))
print(df.info())
#print(df.isnull().sum())
print(df.nunique())

df.dropna(subset=['Postal Code'], inplace=True)
print(df.isnull().sum())
#new column
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
df['Ship_Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
#new 3 date columns
df['Order_Year'] = df['Order Date'].dt.year
df['Order_Month'] = df['Order Date'].dt.month_name()
df['Order_DayOfWeek'] = df['Order Date'].dt.day_name()

# customer classification
customer_sales = df.groupby('Customer ID')['Sales'].transform('sum')
df['Customer_Value'] = pd.qcut(customer_sales, 3, labels=["Low", "Mid", "High"])
print(df.head(2))
# 5. Saqlash
df.to_csv('Train_transformed.csv', index=False)
print("Yakunlandi")