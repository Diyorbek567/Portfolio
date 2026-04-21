import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
# Faylni yuklash va dastlabki tozalash
file=pd.read_excel('отчет по продажам1.xlsx',sheet_name=0,header=3,usecols=[0,1,3,5,6,7,8,9,11,15,16])
# print(file)
# print(file.columns.values,file.dtypes,file)
# Vaqt ustunlarini qo'shish
file['Период'] = pd.to_datetime(file['Период'], errors='coerce')
file = file[~((file['Период'].dt.year == 2025) & (file['Период'].dt.month == 1))]
file = file.dropna(subset=['Период'])
#x
file['Year'] = file['Период'].dt.year
#x
file['Month'] = file['Период'].dt.month
#x
file['year-month'] = file['Период'].dt.to_period('M')
#print('nanlar soni')
#print(file['year-month'].isna().sum())
file[['En', 'Boy']] = file['Размер'].str.split('X', expand=True)
file = file.dropna(subset=['En', 'Boy'])
file['Boy'] = pd.to_numeric(file['Boy'], errors='coerce')
file = file.dropna(subset=['Boy'])
#print(file['Boy'])
file['Количество метров'] = file['Количество метров'].replace({'\xa0': '', ',': '.'}, regex=True)
file['Количество метров'] = file['Количество метров'].apply(lambda x: float(x) if pd.notnull(x) and x != "" else x)
file['quantity'] = file['Количество метров'] /(file['Boy'] * 0.01)
#print(file['quantity'])
file.drop(columns=['En','Boy'],inplace=True)
#y
file['quantity'].fillna(file['Количество штук'], inplace=True)
#print(file.dtypes,file['quantity'])
min_date = file["year-month"].min().to_timestamp()  # Eng eski sana
max_date = file["year-month"].max().to_timestamp()  # Eng yangi sana
# Sanalar oralig‘idagi barcha oylarni olish
dates = pd.date_range(start=min_date, end=max_date, freq='MS').strftime('%Y-%m')
#Har bir nomenklatura uchun MultiIndex yaratish
nomenclatures = file["Номенклатура"].unique()
multi_index = pd.MultiIndex.from_product([nomenclatures, dates], names=["Номенклатура", "year-month"])

# **📌 Barcha ustunlarni guruhlash**
file = file.set_index(["Номенклатура", "year-month"])

#CLEANING
remove_words = ["GREY", "BEIGE", "GREEN", "BLUE", "BEJ"]
# 'Коллекция' ustunidagi ma'lumotlarni to'g'irlash
def clean_collection(value):
    if isinstance(value, str):  
        # "_K", "_M" va "." ni olib tashlash
        value = value[:-2] if value.endswith('_K') or value.endswith('_M') else value
        value = value[:-1] if value.endswith('.') else value
        
        # Keraksiz so'zlarni olib tashlash
        value = re.sub(r'\b(' + '|'.join(remove_words) + r')\b', '', value).strip()
        
        return value
    return value

# To'g'irlangan ma'lumotlarni 'Коллекция' ustuniga qo'llash
file['Коллекция'] = file['Коллекция'].apply(clean_collection)
#cumma
file['Сумма'] = file['Сумма'].astype(str).replace({r'\s': '', ',': '.'}, regex=True).astype(float)
file['Сумма'] = pd.to_numeric(file['Сумма'], errors='coerce')

#print(file['Сумма'])
file['Количество, м2'] = pd.to_numeric(file['Количество, м2'], errors='coerce')
file['Количество, м2']=file['Количество, м2'].fillna(file['Количество, м2'].mean())
#print(file['Сумма'].isnull().sum())
#rint(file['Количество, м2'].isnull().sum())
# 'price' ustunini yaratish
file['price'] = file['Сумма'] / file['Количество, м2']
file['price'] = file['price'].apply(lambda x: 700000 if x > 700000 else (40000 if x < 40000 else x))
#file['xn']=file['price']
#print(file[file['price'] == file['price'].max()])
scaler = MinMaxScaler()
#x
file['price'] = scaler.fit_transform(file[['price']])
#print(file['price'])
#file.to_excel('123.xlsx',sheet_name='Forecast',index=False

file['Себестоимость'] = pd.to_numeric(file['Себестоимость'], errors='coerce')
file['Себестоимость']=file['Себестоимость'].fillna(file['Себестоимость'].mean())
#print(file['Себестоимость'])
file['Сумма, USD'] = pd.to_numeric(file['Сумма, USD'], errors='coerce')
file['Сумма, USD']=file['Сумма, USD'].fillna(file['Сумма, USD'].mean())
#print(file['Сумма, USD'])
#x 
file['Себестоимость']=file['Себестоимость']/file['Сумма, USD']
#print(file['Себестоимость'])
#x
file['nomenklatura']=file['quantity']
# # So'nggi oy va kelasi oyni aniqlash
last_year = file['Year'].max()
last_month = file[file['Year'] == last_year]['Month'].max()
next_month = 1 if last_month == 12 else last_month + 1
next_year = last_year if next_month != 1 else last_year + 1
file.drop(columns=['Количество штук','Период','Количество, м2','Количество метров','Сумма, USD','Сумма'],inplace=True)

#davomi
# **📌 Faqat sonli ustunlarni yig‘indisini olish**
numeric_cols = file.select_dtypes(include=['number']).columns
file_grouped = file.groupby(["Номенклатура", "year-month"])[numeric_cols].sum(min_count=1)

# **📌 Sonli bo'lmagan ustunlarni saqlash (oxirgi qiymatni olish)**
other_cols = [col for col in file.columns if col not in numeric_cols]
file_other = file.groupby(["Номенклатура", "year-month"])[other_cols].last()

# **📌 Hammasini birlashtirish**
file_grouped = file_grouped.join(file_other)

# **MultiIndex bo‘yicha moslashtirish**
file_grouped = file_grouped.reindex(multi_index).reset_index()

# **"quantity" 0 bo‘lsa, boshqa ustunlarga NaN qo‘yish**
for col in numeric_cols:
    if col != "nomenklatura":
        file_grouped[col] = file_grouped[col].where(file_grouped["nomenklatura"] > 0, other=pd.NA)
file_grouped["Year"] = file_grouped["year-month"].str.split("-").str[0].astype(int)
file_grouped["Month"] = file_grouped["year-month"].str.split("-").str[1].astype(int)
# 'Номенклатура' bo‘yicha guruhlab, bo‘sh qiymatlarni to‘ldirish
cols_to_fill = ["Дизайн", "Коллекция", "Размер"]
file_grouped[cols_to_fill] = file_grouped.groupby("Номенклатура")[cols_to_fill].ffill().bfill()
file_grouped[["price", "Себестоимость", "nomenklatura"]] = file_grouped[["price", "Себестоимость", "nomenklatura"]].fillna(0)
file_grouped = file_grouped[["Номенклатура", "Коллекция", "Дизайн", "Размер", "year-month", "Year", "Month", "nomenklatura",'Себестоимость','price','quantity']]
file=file_grouped
file = file.reset_index()  
#Natijani Excelga saqlash
#file_grouped.to_excel("1234.xlsx",index=False)
#print('tamam tamam')
#sezon
file['sezon'] = file['Month'].apply(lambda x: 'low' if x in [1,2,3,4]
                                 else 'mid' if x in [5,6,7,12]
                                 else 'high')
#x
#file['season_quantity'] = file.groupby(['Номенклатура', 'sezon'])['quantity'].transform('sum')
#encoder
encoder = OneHotEncoder(sparse_output=False,dtype=int)
#x
encoded = encoder.fit_transform(file[['sezon']])
columns = encoder.get_feature_names_out(['sezon'])
file[columns]=encoded
#encoded_df = pd.DataFrame(encoded, columns=columns)
#print(encoded_df.isna().sum(), encoded_df.value_counts())
#file = pd.concat([file, encoded_df], axis=1).drop(columns=['sezon'])
file.drop(columns=['sezon'], inplace=True)
file = file.dropna(subset=['Дизайн'])
#print(file.isna().sum(),file.columns)
#print(file)
# Yillik trend (Yearly_Trend)
# Dizayn va vaqt bo‘yicha saralash
#print("NaT qiymatlar soni:", file['year-month'].isna().sum())
latest_months = file['year-month'].dropna().sort_values().unique()[-4:-1]  # Oxirgi 3 oy
#print(latest_months)

# Oxirgi 3 oy uchun har bir `Дизайн` bo‘yicha quantity ni hisoblash
design_last_3_months = (
    file[file['year-month'].isin(latest_months)]
    .groupby(['Дизайн', 'year-month'])['quantity']
    .sum()
    .unstack(fill_value=0)  # Yo‘q bo‘lgan oylarga 0 qo‘yish
)
# **Oxirgi 3 oyda bo‘lmagan dizaynlar uchun 0 qo‘yish**
all_designs = file['Дизайн'].unique()  # Barcha dizaynlar ro‘yxati
design_last_3_months = design_last_3_months.reindex(all_designs, fill_value=0)
# Oxirgi 3 oylik sotuvlar yig‘indisini hisoblash va 3 ga bo‘lish
design_last_3_months['design_sales_mean'] = design_last_3_months.sum(axis=1) / 3

# **Olingan qiymatlarni asosiy `file` DataFrame ga qo‘shish**
file['design_sales_mean'] = file['Дизайн'].map(design_last_3_months['design_sales_mean'])

# Oxirgi 3 oy uchun har bir Коллекция bo‘yicha quantity ni hisoblash
# collection_last_3_months = (
#     file[file['year-month'].isin(latest_months)]
#     .groupby(['Коллекция', 'year-month'])['quantity']
#     .sum()
#     .unstack(fill_value=0)  # Agar ma'lumot bo'lmasa, 0 qo‘yiladi
# )
# # **Oxirgi 3 oyda bo‘lmagan kolleksiyalar uchun 0 qo‘yish**
# all_collections = file['Коллекция'].unique()  # Barcha kolleksiyalar ro‘yxati
# collection_last_3_months = collection_last_3_months.reindex(all_collections, fill_value=0)  # Bo‘lmaganlarga 0 qo‘yish
# # Oxirgi 3 oy o'rtachasini hisoblash
# collection_last_3_months['collection_sales_mean'] = collection_last_3_months.mean(axis=1)/3
# #x
# # Olingan qiymatlarni asosiy `file` DataFrame ga qo‘shish
# file['collection_sales_mean'] = file['Коллекция'].map(collection_last_3_months['collection_sales_mean'])
## `Коллекция` va `year-month` bo‘yicha saralash
# Avval eski tartibni saqlab qolamiz
file['original_index'] = file.index  

# year-month ustunini datetime formatga o‘tkazish
#file['year-month'] = pd.to_datetime(file['year-month'])

# Har bir Коллекция va year-month bo‘yicha quantity yig‘indisini hisoblash
monthly_totals = (
    file.groupby(['Коллекция', 'year-month'])['quantity']
    .sum()
    .reset_index()
)

# Har bir qator uchun oldingi 3 oy o‘rtacha sotuvini hisoblash
def get_past_3_months_avg(collection, current_month):
    prev_3_months = monthly_totals[
        (monthly_totals['Коллекция'] == collection) &
        (monthly_totals['year-month'] < current_month)
    ].sort_values(by='year-month').tail(3)  # Faqat oxirgi 3 oyni olish

    if prev_3_months.empty:
        return 0  # Agar oldingi oylar bo‘lmasa, 0 qaytariladi

    return prev_3_months['quantity'].sum() / len(prev_3_months)  # Faqat mavjud oylar soniga bo‘lish

# 'collection_sales_mean' ustunini qo‘shish
file['collection_sales_mean'] = file.apply(
    lambda row: get_past_3_months_avg(row['Коллекция'], row['year-month']), axis=1
)
# Eski tartibni qaytarish
file = file.sort_values(by='original_index').drop(columns=['original_index'])


# `year-month` ustuni Period formatida bo'lsa, uni datetime64 ga o'tkazamiz
if isinstance(file['year-month'].dtype, pd.PeriodDtype):
    file['year-month'] = file['year-month'].dt.to_timestamp()

else:
    file['year-month'] = pd.to_datetime(file['year-month'], format='%Y-%m')
file['prev_year-month'] = file['year-month'] + pd.DateOffset(months=1)
file['prev_year-month_2'] = file['year-month'] + pd.DateOffset(months=2)
file['prev_year-month_3'] = file['year-month'] + pd.DateOffset(months=3)
# Har bir dizayn va oldingi oy bo‘yicha quantity summasini hisoblaymiz
monthly_sum = file.groupby(['Дизайн', 'prev_year-month'])['quantity'].sum()
monthly_sum_2 = file.groupby(['Дизайн', 'prev_year-month_2'])['quantity'].sum()
monthly_sum_3 = file.groupby(['Дизайн', 'prev_year-month_3'])['quantity'].sum()
# Hisoblangan qiymatlarni o'z joyiga qo‘yamiz
file['lag1'] = file.set_index(['Дизайн', 'year-month']).index.map(monthly_sum).fillna(0)
file['lag2'] = file.set_index(['Дизайн', 'year-month']).index.map(monthly_sum_2).fillna(0)
file['lag3'] = file.set_index(['Дизайн', 'year-month']).index.map(monthly_sum_3).fillna(0)
# Har bir nomenklatura va oldingi oy bo‘yicha quantity summasini hisoblaymiz
monthly_sum = file.groupby(['Номенклатура', 'prev_year-month'])['quantity'].sum()
monthly_sum_2 = file.groupby(['Номенклатура', 'prev_year-month_2'])['quantity'].sum()
monthly_sum_3 = file.groupby(['Номенклатура', 'prev_year-month_3'])['quantity'].sum()
# Hisoblangan qiymatlarni o'z joyiga qo‘yamiz
file['lag1_n'] = file.set_index(['Номенклатура', 'year-month']).index.map(monthly_sum).fillna(0)
file['lag2_n'] = file.set_index(['Номенклатура', 'year-month']).index.map(monthly_sum_2).fillna(0)
file['lag3_n'] = file.set_index(['Номенклатура', 'year-month']).index.map(monthly_sum_3).fillna(0)
file['year-month'] = file['year-month'].dt.strftime('%Y-%m')
# file['dizayn_oylik_quantity']=file.groupby(['Дизайн','year-month'])['quantity'].transform('sum')
# file['lag1'] = file.groupby('Дизайн').apply(lambda x: x.set_index('year-month')['dizayn_oylik_quantity'].shift().reset_index()['dizayn_oylik_quantity']).reset_index(drop=True)
#print(file[['Дизайн', 'year-month', 'dizayn_oylik_quantity']].dropna().head(10))
#file['lag1'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(1).fillna(0)
# file['lag2'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(2).fillna(0)
# file['lag3'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(3).fillna(0)
# # # Dizayn bo‘yicha alohida shift qilish va yangi ustunlarni yaratish
# # file['lag_1'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(1)
# # file['lag_2'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(2)
# # file['lag_3'] = file.groupby('Дизайн')['dizayn_oylik_quantity'].shift(3)
# print(file['lag1'])
# print(file['lag2'])
# print(file['lag3'])
# Natijani asosiy `file` dataframe'ga qo'shish**
#file = file.merge(monthly_sales, on=['Дизайн', 'year-month'], how='left')
#features
#features=file[['quantity','Себестоимость','collection_sales_mean','design_sales_mean',
               #'price','Year','Month','year-month','Номенклатура','Дизайн','Коллекция','nomenklatura']]

# all_months = file['year-month'].dropna().sort_values().unique()                                                                #  oylar uchun har bir Номенклатура bo‘yicha quantity ni hisoblash 
# months_quantity = ( 
#     file[file['year-month'].isin(all_months)] 
#     .groupby(['Номенклатура', 'year-month'])['quantity'] 
#     .sum()
#     .unstack(fill_value=0)  # Yo‘q bo‘lgan oylarga 0 qo‘yish
# )

# all_n = file['Номенклатура'].unique()  # Barcha n ro‘yxati 
# months_quantity = months_quantity.reindex(index=all_n, fill_value=0) 
# months_quantity.to_excel('12345.xlsx',sheet_name='Forecast',index=True)
# print('tamom shut')
# file.drop(columns=['Количество метров','Количество штук','Количество, м2','Сумма','Сумма, USD','Boy','prev_year-month','prev_year-month_2','prev_year-month_3'],inplace=True)
# file['year-month'].unique()
# file['Номенклатура'].unique()
# file['nomenklatura']=( 
#     file[file['year-month'].isin(all_months)] 
#     .groupby(['Номенклатура', 'year-month'])['quantity'] 
#     .sum()
#     .unstack(fill_value=0)  # Yo‘q bo‘lgan oylarga 0 qo‘yish
# )

# file.set_index(['Номенклатура'], inplace=True)
file.drop(columns=['quantity','prev_year-month','prev_year-month_2','prev_year-month_3'], inplace=True)
print(file.columns.values,file.columns.dtype)
print(file['Размер'])
#file.to_excel('123.xlsx',sheet_name='Forecast',index=False)
#print('tamom shut')
#print(file['season'].head(30))
#print(file.dtypes)
#correlation
# Korrelyatsiya matritsasini yaratish
# features=file.drop(columns=['year-month','Номенклатура','Дизайн','Коллекция','Размер'])
# corr_matrix = features.corr()

# # Konsolga chiqarish
# print("Korrelyatsiya matritsasi:\n", corr_matrix)

# # Issiqlik xaritasi orqali vizuallashtirish
# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Korrelyatsiya Matritsasi")
# plt.show()
file.drop(columns=['Номенклатура','index','Коллекция'],inplace=True)
#prognozlash
# print(file['Размер'])
razmer = list(file['Размер'].unique())
print(file.iloc[0])
forecasts = []

# "Boy" ustunini "Размер" ustunidan ajratib olish
file['Boy'] = file['Размер'].apply(lambda x: int(x.split('X')[1]))

# Donalik va metrajni aniqlash
file['Unit'] = np.where(file['Boy'] > 1000, 'metraj', 'dona')

count = 1
for current_size in razmer:
    print(len(razmer) - count)
    count += 1
    size_group = file[file['Размер'] == current_size]
    for dizayn, guruh in size_group.groupby('Дизайн'):
        filtered_data = guruh.sort_values(by='year-month').copy()
        
        # Donalik va metraj uchun alohida modellar
        unit_type = filtered_data['Unit'].iloc[-1]
        
        train_set = filtered_data[filtered_data['year-month'] < '2024-12']
        test_set = filtered_data[filtered_data['year-month'] >= '2024-12']

        X_train = train_set.drop(columns=['year-month', 'Размер', 'nomenklatura', 'Дизайн', 'Boy'])
        X_test = test_set.drop(columns=['year-month', 'Размер', 'nomenklatura', 'Дизайн', 'Boy'])
        y_train = train_set['nomenklatura']
        y_test = test_set['nomenklatura']
        
        # "Unit" ustunini one-hot encoding qilish
        X_train = pd.get_dummies(X_train, columns=['Unit'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['Unit'], drop_first=True)
        feature_order = list(X_train.columns)

        # Standartlashtirish
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        
        if unit_type == 'dona':
            # Sotiladimi yoki yo‘q? (Classification)
            y_train_class = (y_train > 0).astype(int)
            y_test_class = (y_test > 0).astype(int)

            class_model = RandomForestClassifier(n_estimators=100, random_state=42)
            class_model.fit(X_train_scaled, y_train_class)
            
            # Sotiladigan mahsulotlar uchun qancha sotiladi? (Regression)
            train_idx = y_train_class[y_train_class == 1].index.intersection(X_train_scaled.index)
            
            if not train_idx.empty:
                reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
                reg_model.fit(X_train_scaled.loc[train_idx], y_train.loc[train_idx])
            
            # Kelasi oy prognozi
            next_month_data = pd.DataFrame([{**{
                'Себестоимость': filtered_data['Себестоимость'].iloc[-1],
                'price': filtered_data['price'].iloc[-1],
                'Year': next_year,
                'Month': next_month,
                'sezon_high': filtered_data['sezon_high'].iloc[-1],
                'sezon_low': filtered_data['sezon_low'].iloc[-1],
                'sezon_mid': filtered_data['sezon_mid'].iloc[-1],
                'collection_sales_mean': filtered_data['collection_sales_mean'].iloc[-1],
                'design_sales_mean': filtered_data['design_sales_mean'].iloc[-1],
                'lag1': filtered_data['lag1'].iloc[-1],
                'lag2': filtered_data['lag2'].iloc[-1],
                'lag3': filtered_data['lag3'].iloc[-1],
                'lag1_n': filtered_data['lag1_n'].iloc[-1],
                'lag2_n': filtered_data['lag2_n'].iloc[-1],
                'lag3_n': filtered_data['lag3_n'].iloc[-1]
            }}])

            next_month_data = next_month_data.reindex(columns=feature_order)
            next_month_data_scaled = scaler.transform(next_month_data)

            will_sell = class_model.predict(next_month_data_scaled)[0]
            next_month_forecast = round(reg_model.predict(next_month_data_scaled)[0]) if will_sell == 1 else 0
        
        else:
            # Metraj uchun oddiy regressiya modeli
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            next_month_data = pd.DataFrame([{**{
                'Себестоимость': filtered_data['Себестоимость'].iloc[-1],
                'price': filtered_data['price'].iloc[-1],
                'Year': next_year,
                'Month': next_month,
                'sezon_high': filtered_data['sezon_high'].iloc[-1],
                'sezon_low': filtered_data['sezon_low'].iloc[-1],
                'sezon_mid': filtered_data['sezon_mid'].iloc[-1],
                'collection_sales_mean': filtered_data['collection_sales_mean'].iloc[-1],
                'design_sales_mean': filtered_data['design_sales_mean'].iloc[-1],
                'lag1': filtered_data['lag1'].iloc[-1],
                'lag2': filtered_data['lag2'].iloc[-1],
                'lag3': filtered_data['lag3'].iloc[-1],
                'lag1_n': filtered_data['lag1_n'].iloc[-1],
                'lag2_n': filtered_data['lag2_n'].iloc[-1],
                'lag3_n': filtered_data['lag3_n'].iloc[-1]
            }}])
            
            next_month_data = next_month_data.reindex(columns=feature_order)
            next_month_data_scaled = scaler.transform(next_month_data)
            next_month_forecast = model.predict(next_month_data_scaled)[0]
        
        real_nomenklatura = test_set[(test_set['Размер'] == current_size) & (test_set['Дизайн'] == dizayn)]['nomenklatura']
        nomenklatura_value = real_nomenklatura.iloc[0] if not real_nomenklatura.empty else None
        
        forecasts.append((current_size, dizayn, next_month_forecast, nomenklatura_value))

print("Yaxshi ketyapti")

# Natijalarni DataFramega aylantirish
forecast_df = pd.DataFrame(forecasts, columns=['Размер', 'Дизайн', 'Next_Month_Forecast','Real_Nomenklatura'])
sales = file.groupby('Дизайн')['nomenklatura'].sum().reset_index()
sales.rename(columns={'nomenklatura': 'Total_Quantity'}, inplace=True)
forecast_df = forecast_df.merge(sales, on='Дизайн', how='left')

# Harflar va raqamlarni ajratib olish funksiyalari
def extract_letters(design):
    return ''.join(re.findall(r'[A-Za-z]', str(design)))

def extract_numbers(design):
    return '_'.join(re.findall(r'\d+', str(design)))

forecast_df['Letters'] = forecast_df['Дизайн'].apply(extract_letters)
forecast_df['Numbers'] = forecast_df['Дизайн'].apply(extract_numbers)

# Rankingni oshish tartibida qilish, har bir 'Letters' uchun
forecast_df['Next_Month_Forecast'] = forecast_df['Next_Month_Forecast'].apply(lambda x:0 if x<0 else x ).round(2)
# 1. Birinchi rank (prognoz bo‘yicha tartiblash)
forecast_df = forecast_df.sort_values(
    by=['Размер', 'Letters', 'Next_Month_Forecast', 'Total_Quantity'], 
    ascending=[True, True, False, False]
)
forecast_df['rank_f'] = forecast_df.groupby(['Размер', 'Letters']).cumcount() + 1

# 2. Ikkinchi rank (haqiqiy sotuvlar bo‘yicha tartiblash)
forecast_df = forecast_df.sort_values(
    by=['Размер', 'Letters', 'Real_Nomenklatura','Total_Quantity'], 
    ascending=[True, True, False,False]
)
forecast_df['rank_real'] = forecast_df.groupby(['Размер', 'Letters']).cumcount() + 1

# Final natijani chiqarish
forecast_df = forecast_df[['Размер','Letters','Numbers', 'rank_f', 'rank_real', 'Next_Month_Forecast', 'Real_Nomenklatura']]

## Rank farqini hisoblash
forecast_df['rank_difference'] = abs(forecast_df['rank_f'] - forecast_df['rank_real'])

# O‘rtacha absolyut xatolik (MAE)
mae = forecast_df['rank_difference'].mean()
print(f"Modelning o‘rtacha xatoligi (MAE): {mae}")

# To‘g‘ri topilgan foiz (Accuracy)
correct_predictions = (forecast_df['rank_difference'] == 0).sum()
total_predictions = len(forecast_df)
accuracy = (correct_predictions / total_predictions) * 100
print(f"Model to‘g‘ri topgan prognozlar: {accuracy:.2f}%")
a=forecast_df['rank_difference'].sum()
b=forecast_df['rank_real'].sum()
c=1-a/b
print(f"Modelning aniqligi-{c}")
# Yakuniy faylni saqlash
output_file = "forecast_evaluation.xlsx"
forecast_df.to_excel(output_file, index=False, sheet_name="Evaluation")

print(f"Natijalar {output_file} fayliga muvaffaqiyatli saqlandi!")



# Modelni baholash
# y_pred = model.predict(X_test_scaled)

# # R², MAE, MSE va RMSE hisoblash
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print(f"R²: {r2:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")