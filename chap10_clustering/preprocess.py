import time
from scipy import stats
import os

# user - products
user_product_dic = {}
# product - users
product_user_dic= {}

# product_id - product_name
product_id_name_dic = {}

# http://archive.ics.uci.edu/ml/machine-learning-databases/00352/
uci_online_retail_dir ='/home/socurites/Backup/Data/UCI'
for line in open(os.path.join(uci_online_retail_dir, 'online_retail.csv')):
    line_items = line.strip().split('\\')
    # 0:    InvoiceNo
    # 1:    StockCode
    # 2:    Description
    # 3:    Quantity
    # 4:    InvoiceDate   MM/DD/YYYY HH:MM      12/1/2010 8:26
    # 5:    UnitPrice
    # 6:    CustomerID
    # 7:    Country

    user_code = line_items[6]
    product_id = line_items[1]
    product_name = line_items[2]
    country = line_items[7]

    if len(user_code) == 0:
        continue

    if country != 'United Kingdom':
        continue


    try:
        invoice_year = time.strptime(line_items[4], "%m/%d/%Y %H:%M").tm_year
    except ValueError:
        continue

    if invoice_year != 2011:
        continue

    user_product_dic.setdefault(user_code, set())
    product_user_dic.setdefault(product_id, set())

    user_product_dic[user_code].add(product_id)
    product_user_dic[product_id].add(user_code)

    product_id_name_dic[product_id] = product_name

# #product per user
product_per_user_li = [len(x) for x in user_product_dic.values()]

print('# of users: %d' % len(user_product_dic))
print('# of products: %d' % len(product_user_dic))

# basic stats per #items per user
print(stats.describe(product_per_user_li))


'''
Plot data distribution
- X: #products per user
- Y: #users
'''

import matplotlib.pyplot as plt
from collections import Counter

plot_data_all = Counter(product_per_user_li)
plot_data_x = list(plot_data_all.keys())            # #products per user
plot_data_y = list(plot_data_all.values())          # #users
plt.xlabel('#products per user')
plt.ylabel('#users')
plt.scatter(plot_data_x, plot_data_y, marker='+')

plt.show()


'''
Remove noise data
'''