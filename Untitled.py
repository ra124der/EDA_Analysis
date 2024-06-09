#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Data Analysis with Streamlit')

# File path
file_path = 'Alpha.csv'

# Read the CSV file into a DataFrame with encoding specified
try:
    df = pd.read_csv(file_path, delimiter=",", on_bad_lines='skip', engine='python', encoding='latin1')
    st.write("Data loaded successfully")
except pd.errors.ParserError as e:
    st.error(f"ParserError: {e}")
except UnicodeDecodeError as e:
    st.error(f"UnicodeDecodeError: {e}")

# Display the dataframe
st.write(df)

# Data shape
st.subheader('Data Shape')
st.write(df.shape)

# Unique values per column
st.subheader('Unique Values per Column')
st.write(df.apply(pd.Series.nunique))

# Missing values per column
st.subheader('Missing Values per Column')
st.write(df.isnull().sum(axis=0))

# Drop columns with only one unique value
cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=cols_to_drop)
st.write(f"Data shape after dropping columns with one unique value: {df.shape}")

# Fill missing values in 'buyers_fees' column with 0
column_name = 'buyers_fees'
default_value = 0
df[column_name] = df[column_name].fillna(default_value)
st.write(f"Filled missing values in '{column_name}' column with {default_value}")
st.write(df[column_name])

# Drop rows where 'product_category' is missing
df = df.dropna(subset=['product_category'])
st.write(f"Data shape after dropping rows with missing 'product_category': {df.shape}")

# Drop specified columns
columns_to_drop = ['product_name', 'product_description', 'product_keywords', 'brand_url', 'seller_username', 'product_color', 'brand_name',
                   'seller_pass_rate', 'seller_num_followers', 'seller_community_rank', 'warehouse_name', 'seller_badge', 'product_material',
                   'should_be_gone']
df = df.drop(columns=columns_to_drop)
st.write(f"Data shape after dropping specified columns: {df.shape}")

# Convert True/False to 1/0 for the specified columns
columns_to_convert = ['sold', 'available', 'in_stock']
df[columns_to_convert] = df[columns_to_convert].astype(int)
st.write("Converted True/False to 1/0 for the specified columns:")
st.write(df[columns_to_convert])

# Plot distribution of top 10 brands
st.subheader('Distribution of Top 10 Popular Brands')
st.write('This bar chart shows the distribution of the top 10 most popular brands in the dataset. '
         'It indicates which brands are most prevalent among the products listed, reflecting brand popularity and consumer preferences.')
brand_counts = df['brand_id'].value_counts()
top_10_brands = brand_counts.nlargest(10).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_brands.index.astype(str), y=top_10_brands.values, palette='viridis')
plt.title('Distribution of Top 10 Popular Brands')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i in range(len(top_10_brands)):
    plt.text(i, top_10_brands.values[i], str(top_10_brands.values[i]), ha='center', va='bottom')
st.pyplot(plt)
st.write("The chart shows that the brand having brand id 2 (Gucci) a significantly higher count, indicating a strong consumer preference for this brand.Brand id 94(Burberry) and Brand id 47(Dolce & Gabbana)are the next most popular brands")
# Plot distribution of top 10 product types
st.subheader('Top 10 Product Types')
st.write('This bar chart illustrates the top 10 product types in the dataset, showing the count of each product type. '
         'This helps identify which types of products are most common and likely in demand.')
product_type_counts = df['product_type'].value_counts()
top_10_product_types = product_type_counts.nlargest(10).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_product_types.index, y=top_10_product_types.values, palette='viridis')
plt.title('Top 10 Product Types')
plt.xlabel('Product Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i in range(len(top_10_product_types)):
    plt.text(i, top_10_product_types.values[i], str(top_10_product_types.values[i]), ha='center', va='bottom')
st.pyplot(plt)
st.write("Sunglasses is the most popular product type followed by jacket,silk tie,t-shirt in second,third and fourth")

# Scatter plot for seller price vs product type
st.subheader('Relationship Between Seller Price and Product Type (Top 20 Product Types)')
st.write('This scatter plot shows the relationship between seller price and product type for the top 20 product types. '
         'A logarithmic scale is used for the y-axis to better visualize price variations. This plot helps identify pricing trends across different product types.')
top_product_types = df['product_type'].value_counts().nlargest(20).index
df_filtered = df[df['product_type'].isin(top_product_types)]
df_sample = df_filtered.sample(n=10000, random_state=42)
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df_sample, x='product_type', y='seller_price', hue='product_type', alpha=0.6, palette='viridis')
plt.title('Relationship Between Seller Price and Product Type (Top 20 Product Types)')
plt.xlabel('Product Type')
plt.ylabel('Seller Price')
plt.yscale('log')
plt.legend(title='Product Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
st.pyplot(plt)
st.write("The scatter plot indicates range of prices within each product type.It can be seen that watches and jackets have a high range of prices from very cheap to very costly.On the other hand,prices of silk ties,shirts and hats do not vary that much.")

# Pie chart for product season distribution
st.subheader('Distribution of Product Season')
st.write('This pie chart depicts the distribution of products across different seasons. '
         'It highlights the seasonal trends and can help understand the seasonality in product listings.')
season_distribution = df['product_season'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(season_distribution, labels=season_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(season_distribution)))
plt.title('Distribution of Product Season')
plt.axis('equal')
st.pyplot(plt)
st.write("The pie chart shows the proportion of products categorized by season. It indicates that most of products sold on the online marketplace alpha are suitable to be used in all seasons.12.7 percent are suitable to be used in autumn/winter season and 4.2 percent are suitable to be used in spring/summer season.")

# Bar chart for top 15 countries with the most sellers
st.subheader('Top 15 Countries with the Most Sellers')
st.write('This bar chart shows the top 15 countries with the most sellers, along with the count of sellers from each country. '
         'This information is useful for understanding the geographical distribution of sellers.')
country_seller_count = df['seller_country'].value_counts()
top_15_countries = country_seller_count.head(15)
plt.figure(figsize=(10, 6))
bars = plt.bar(top_15_countries.index, top_15_countries.values)
plt.title('Top 15 Countries with the Most Sellers')
plt.xlabel('Country')
plt.ylabel('Number of Sellers')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
st.pyplot(plt)
st.write("The bar chart indicates that certain countries have a higher number of sellers in alpha online marketplace.Italy has highest number of sellers in marketplace,followed by France,USA and UK.")







