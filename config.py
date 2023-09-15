 # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json






STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# Content based

df=pd.read_parquet('df.parquet')
df_clean=pd.read_parquet('product_ws.parquet')
df.rename(columns={'item_id': 'product_id','name':'product_name'}, inplace=True)
df_clean.rename(columns={'item_id': 'product_id','name':'product_name'}, inplace=True)
df_sub=df_clean


# Collab 

# df_collab = pd.read_csv('df_collab_50.csv') # edit
df_collab = pd.read_csv('df_review_full_10.csv')
df_product=pd.read_csv('df_product.csv')
df_review_full = pd.read_csv('df_review_full_10.csv')  



# edit
top_products = df_product['product_id'].value_counts().reset_index()
top_products.columns = ['product_id', 'count']

# Sort the values by count in descending order
top_products = top_products.sort_values(by='count', ascending=False)
# top_products = pd.concat([top_products, df_product[['product_id', 'product_name', 'image','price']]], axis=1) # # edit
top_products = pd.merge(top_products, df_product[['product_id', 'product_name', 'image','price']], on='product_id', how='left')
top_products=top_products.head(5)




###### code ko dùng


# df_review_full = df_review.merge(df_product, on="product_id", how="left")

#review_full (dùng để tìm sản phẩm phố biến nhất)
# print(df_review_full.columns)
# print(df_product.columns)














# # # Rename columns
# column_name_mapping = {
#     "item_id": "product_id",
#     "name": "product_name",
#     "rating":"product_rating"
# }
# df_product = df.rename(columns=column_name_mapping)




# # # Convert data types
# df_review['product_id'] = pd.to_numeric(df_review['product_id'], errors='coerce')
# df_review.rename(columns={'rating': 'customer_rating'}, inplace=True)
# # # Merge dataframes
# df_review_full = df_review.merge(df_product, on="product_id", how="left")# df review khi gop vs product_id

# df_review_full.to_csv('df_review_full.csv')
# print(df_review_full.columns)













# df_collab=df_collab.merge(df_review[["product_id","name"]], on=["product_id"], how="left")
# df_collab.rename(columns={'name': 'customer_name'}, inplace=True)

# df_collab = df_collab.drop_duplicates()

# print(df_collab.columns)




# # df_collab=df_collab.merge(df_review_full[["product_id","customer_id","created_time"]], on=["product_id","customer_id"], how="left")
# df_collab=df_collab.merge(df_review[["product_id","name"]], on=["product_id"], how="left")
# df_collab.rename(columns={'name': 'customer_name'}, inplace=True)
# df_collab = df_collab.drop_duplicates()

# print(df_collab.columns)

# df_review=pd.read_csv('ReviewRaw.csv')





# print(top_products.head())



































# save file as parquet
# # Load data
# df_review = pd.read_csv('ReviewRaw.csv', header=0)
# # df_review_sub = pd.read_parquet('df_sub.parquet')
# df_review_result = pd.read_csv('result_df.csv', header=0)
# df_review_result=df_review_result[["customer_id","product_id","rating"]]

# # Rename columns
# column_name_mapping = {
#     "item_id": "product_id",
#     "name": "product_name",
# }
# df_product = df.rename(columns=column_name_mapping)

# # # Convert data types
# df_review['product_id'] = pd.to_numeric(df_review['product_id'], errors='coerce')

# # # Merge dataframes
# #df_review_full = df.merge(df_product, on="product_id", how="left")
# df_collab2=df_review_result.merge(df_product, on="product_id", how="left")

# print(df_collab2.columns)
# df_collab2=df_collab2.drop(columns = ['url', 'rating_y'])
# df_collab2 = df_collab2.rename(columns={'rating_x': 'customer_rating'})

# df_collab2.to_csv('df_collab2.csv')

