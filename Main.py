import math
import sys
import getopt
import cv2
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Main 
#import CSV files to data frames

orders =  pd.read_csv("olist_orders_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
order_payments = pd.read_csv("olist_order_payments_dataset.csv")
order_items_details = pd.read_csv("olist_order_items_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")

# take only delivered products for consideration
orders = orders[orders['order_status'] == 'delivered'] 

# creating one needed / workable dataset from all the set 
total_orders = orders.merge(customers,on = 'customer_id')
total_orders =  total_orders.merge(order_reviews,on="order_id")
total_orders.drop_duplicates(subset ="order_id", keep = False, inplace = True)
total_orders = total_orders.merge(order_payments,on="order_id")
total_orders.drop_duplicates(subset ="order_id", keep = False, inplace = True)
total_orders = total_orders.merge(order_items_details,on="order_id")
total_orders.drop_duplicates(subset ="order_id", keep = False, inplace = True)
total_orders = total_orders.merge(sellers,on="seller_id")
total_orders = total_orders.merge(products,on="product_id")

# write the merged data frame to csv so we dont need to run again 
#total_orders.to_csv("merged_total_order.csv")

#total_orders = pd.read_csv("merged_total_order.csv")

# converting data type to date time (it was string before)
total_orders['order_purchase_timestamp'] = pd.to_datetime(total_orders['order_purchase_timestamp'])
total_orders['order_delivered_customer_date'] = pd.to_datetime(total_orders['order_delivered_customer_date'])
total_orders['order_approved_at'] = pd.to_datetime(total_orders['order_approved_at'])
total_orders['order_estimated_delivery_date'] = pd.to_datetime(total_orders['order_estimated_delivery_date'])

#create new attributes from existing attributes
total_orders['delivery_month'] = total_orders['order_estimated_delivery_date'].dt.strftime('%m/%Y')
total_orders['delevery_days'] = (total_orders['order_delivered_customer_date'] - total_orders['order_purchase_timestamp']).dt.days;

del_bucket = []
for x in total_orders['delevery_days']:
    if x < 5:
        del_bucket.append("<5")
    elif x < 10:
        del_bucket.append("5-10")
    elif x < 20:
        del_bucket.append("10-20")
    elif x < 40:
        del_bucket.append("20-40")
    else:
        del_bucket.append(">40")
                   
total_orders['del_time_range'] = del_bucket 

# analyse data if we want to -- pending for the report will do later



total_orders_req_col = total_orders.drop(columns=["order_status","order_delivered_carrier_date","order_purchase_timestamp",
           "order_approved_at","customer_zip_code_prefix","customer_unique_id",
           "customer_city","review_id","review_comment_title","review_comment_message",
           "review_creation_date","review_answer_timestamp","order_item_id","seller_zip_code_prefix",
           "seller_city","product_category_name","delivery_month","product_id","seller_id",
           "order_id","customer_id","order_delivered_customer_date","order_estimated_delivery_date",
           "customer_state","seller_state","payment_sequential","payment_value","del_time_range","payment_type","product_weight_g",
           "product_height_cm","product_length_cm","product_width_cm","payment_installments"])
    
total_orders_req_col["delevery_days"] = pd.to_numeric(total_orders_req_col["delevery_days"])
total_orders_req_col = total_orders_req_col.fillna(0)
train=total_orders_req_col.sample(frac=0.8,random_state=200)
test=total_orders_req_col.drop(train.index)


#f, ax = plt.subplots(figsize=(10, 8))
#corr = total_orders_req_col.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
 #           square=True, ax=ax)

Y_test =  test[test.columns[0]]
X_test =  test[test.columns[1:7]]

Y_train =  train[train.columns[0]]
X_train =  train[train.columns[1:7]]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)
accuracy = regressor.score(X_test,Y_test)
print("Linear model accuracy is = " + str(accuracy*100),'%')
#plt.scatter(Y_test,predictions)
#plt.plot(X_test, regressor.predict(X_test), color='red',linewidth=3)
#plt.scatter(X_test, Y_test)
#plt.plot(Y_test,predictions)
#plt.show()
#print(predictions)

# new derived data frame 

new_total_orders = total_orders
new_total_orders["estimated_del_days"] = (total_orders['order_delivered_customer_date'] - total_orders['order_approved_at']).dt.days;
new_total_orders["delta_time"] =  new_total_orders["estimated_del_days"]-new_total_orders["delevery_days"]

islate = []
for x in total_orders['delta_time']:
    if x < 0:
        islate.append(1)
    else:
        islate.append(0)
new_total_orders["is_late"] =  islate
new_total_orders["total_price"]  = new_total_orders["price"] +new_total_orders["freight_value"]
new_total_orders["freight_ratio"] = new_total_orders["freight_value"]/new_total_orders["price"]
new_total_orders["purchase_day_of_week"] = new_total_orders["order_approved_at"].dt.dayofweek

new_total_orders = new_total_orders.drop(columns=["order_status","order_delivered_carrier_date","order_purchase_timestamp",
           "order_approved_at","customer_zip_code_prefix","customer_unique_id",
           "customer_city","review_id","review_comment_title","review_comment_message",
           "review_creation_date","review_answer_timestamp","order_item_id","seller_zip_code_prefix",
           "seller_city","product_category_name","delivery_month","product_id","seller_id",
           "order_id","customer_id","order_delivered_customer_date","order_estimated_delivery_date",
           "customer_state","seller_state","payment_sequential","payment_value","del_time_range","payment_type","product_weight_g",
           "product_height_cm","product_length_cm","product_width_cm","payment_installments"])


new_total_orders["delevery_days"] <- pd.to_numeric(new_total_orders["delevery_days"])
new_total_orders["delta_time"] =  pd.to_numeric(new_total_orders["delta_time"])
new_total_orders["estimated_del_days"] =  pd.to_numeric(new_total_orders["estimated_del_days"])

new_review_score = []
for x in total_orders['review_score']:
    if x < 1.5:
        new_review_score.append(0)
    elif x< 3.5:
        new_review_score.append(1)
    else:
        new_review_score.append(2)

new_total_orders["review_score"] =  new_review_score
new_total_orders["review_score"] =  pd.to_numeric(new_total_orders["review_score"])
new_total_orders = new_total_orders.fillna(0)


#f, ax = plt.subplots(figsize=(10, 8))
#corr = new_total_orders.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)

# apply all the models on the new data set 
train=new_total_orders.sample(frac=0.8,random_state=200)
test=new_total_orders.drop(train.index)

Y_test =  test[test.columns[0]]
X_test =  test[test.columns[1:13]]

Y_train =  train[train.columns[0]]
X_train =  train[train.columns[1:13]]

# linear regression 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
predictions = regressor.predict(X_test)
accuracy = regressor.score(X_test,Y_test)
print("Linear model accuracy is = " + str(accuracy*100),'%')

# logistic regression
new_total_orders['review_score'] = new_total_orders['review_score'].astype('category')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

accuracy = classifier.score(X_test,Y_test)
print("Logistic model accuracy is = " + str(accuracy*100),'%')
