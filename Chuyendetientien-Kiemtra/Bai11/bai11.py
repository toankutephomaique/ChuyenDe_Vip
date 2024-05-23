result = '''
1. id: Mã định danh của ngôi nhà (kiểu số).
2. date: Ngày bán ngôi nhà (kiểu chuỗi). Định dạng có thể là dd/mm/yyyy hoặc mm/dd/yyyy.
3. price: Giá bán của ngôi nhà (mục tiêu dự đoán, kiểu số).
4. bedrooms: Số phòng ngủ trong ngôi nhà (kiểu số).
5. bathrooms: Số phòng tắm trong ngôi nhà (có thể bao gồm cả phòng tắm và phòng vệ sinh, kiểu số).
6. sqft_living: Diện tích sống của ngôi nhà tính bằng feet vuông (kiểu số).
7. sqft_lot: Diện tích lô đất của ngôi nhà tính bằng feet vuông (kiểu số).
8. floors: Tổng số tầng của ngôi nhà (kiểu số).
9. waterfront: Nhà có tầm nhìn ra sông hoặc biển (kiểu số, 0 hoặc 1).
10. view: Số lần nhà đã được xem xét (kiểu số).
11. condition: Điều kiện tổng quát của ngôi nhà, với 1 là rất kém và 5 là rất tốt (kiểu số). Thông tin thêm có thể xem ở Glossary.
12. grade: Đánh giá tổng quát về ngôi nhà theo hệ thống đánh giá của King County, từ 1 là rất kém đến 13 là rất tốt (kiểu số).
13. sqft_above: Diện tích phần trên mặt đất của ngôi nhà tính bằng feet vuông (không bao gồm diện tích tầng hầm, kiểu số).
14. sqft_basement: Diện tích tầng hầm của ngôi nhà tính bằng feet vuông (kiểu số).
15. yr_built: Năm xây dựng ngôi nhà (kiểu số).
16. yr_renovated: Năm ngôi nhà được cải tạo (kiểu số, nếu chưa cải tạo sẽ là 0).
17. zipcode: Mã bưu chính của khu vực ngôi nhà (kiểu số).
18. lat: Vĩ độ của ngôi nhà (tọa độ địa lý, kiểu số).
19. long: Kinh độ của ngôi nhà (tọa độ địa lý, kiểu số).
20. sqft_living15: Diện tích phòng khách tính bằng feet vuông vào năm 2015 (ngụ ý rằng có thể đã có cải tạo).
21. sqft_lot15: Diện tích lô đất tính bằng feet vuông vào năm 2015 (ngụ ý rằng có thể đã có cải tạo).

có 21 trường và 21613  mẫu

             price  bedrooms  bathrooms  sqft_living      sqft_lot    floors
Min   7.800000e+04    1.0000   0.500000    370.00000  5.200000e+02  1.000000
Max   7.700000e+06   33.0000   8.000000  13540.00000  1.651359e+06  3.500000
Mean  5.402966e+05    3.3732   2.115826   2080.32185  1.509941e+04  1.494096

Số lượng nhà có giá dưới 30000 và có 3 phòng ngủ: 0
Số lượng nhà có giá dưới 40000 và có 3 phòng ngủ, 3 phòng tắm: 0
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def writeFile(filepath, data):
    with open(file=filepath, mode='w+', encoding='utf-8', errors='ignore') as f:
        f.write(data)

# Load dữ liệu
data = pd.read_csv(r"D:\Chuyendetientien-Kiemtra\Bai10\kc_house_data.csv")

#Bai 2
# Lọc các cột cần tính toán
cols_to_analyze = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']

# Lọc các cột có kiểu dữ liệu số
numeric_cols = data[cols_to_analyze].select_dtypes(include=['number']).columns

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': data[numeric_cols].min(),
                               'Max': data[numeric_cols].max(),
                               'Mean': data[numeric_cols].mean()},
                              index=numeric_cols)

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)


#Bài 3

# a. Số lượng nhà có giá dưới 30000 và có 3 phòng ngủ
condition_a = (data['price'] < 30000) & (data['bedrooms'] == 3)
count_a = data[condition_a].shape[0]

# b. Số lượng nhà có giá dưới 40000 và có 3 phòng ngủ, 3 phòng tắm
condition_b = (data['price'] < 40000) & (data['bedrooms'] == 3) & (data['bathrooms'] == 3)
count_b = data[condition_b].shape[0]

print(f"Số lượng nhà có giá dưới 30000 và có 3 phòng ngủ: {count_a}")
print(f"Số lượng nhà có giá dưới 40000 và có 3 phòng ngủ, 3 phòng tắm: {count_b}")


writeFile(r"D:\Chuyendetientien-Kiemtra\Bai10\ketqua10.txt", result)

# Bài 4


# Tạo lưới 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Danh sách các màu sẽ sử dụng cho từng biểu đồ
colors = ['r', 'g', 'b', 'm']

# a. price với bedrooms
axs[0, 0].scatter(data['bedrooms'], data['price'], color=colors[0], alpha=0.5)
axs[0, 0].set_title('Price vs Bedrooms')
axs[0, 0].set_xlabel('Bedrooms')
axs[0, 0].set_ylabel('Price')

# b. price với bathrooms
axs[0, 1].scatter(data['bathrooms'], data['price'], color=colors[1], alpha=0.5)
axs[0, 1].set_title('Price vs Bathrooms')
axs[0, 1].set_xlabel('Bathrooms')
axs[0, 1].set_ylabel('Price')

# c. price với sqft_living
axs[1, 0].scatter(data['sqft_living'], data['price'], color=colors[2], alpha=0.5)
axs[1, 0].set_title('Price vs Sqft Living')
axs[1, 0].set_xlabel('Sqft Living')
axs[1, 0].set_ylabel('Price')

# d. sqft_living với floors
axs[1, 1].scatter(data['floors'], data['sqft_living'], color=colors[3], alpha=0.5)
axs[1, 1].set_title('Sqft Living vs Floors')
axs[1, 1].set_xlabel('Floors')
axs[1, 1].set_ylabel('Sqft Living')

# Điều chỉnh layout cho đẹp hơn
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()


