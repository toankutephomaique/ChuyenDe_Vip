result = '''
1. id
2. address: Địa chỉ đầy đủ của căn hộ.
3. city: Thành phố nơi căn hộ đó nằm, có thể là Warsaw, Cracow hoặc Poznan.
4. floor: Số tầng mà căn hộ đó đặt ở.
5. id: ID của căn hộ.
6. latitude: Vĩ độ của căn hộ.
7. longitude: Kinh độ của căn hộ.
8. price: Giá của căn hộ tính bằng PLN (zloty Ba Lan) - đây là mục tiêu mà chúng ta muốn dự đoán.
9. rooms: Số phòng trong căn hộ.
10. sq: Diện tích của căn hộ tính bằng mét vuông.
11. year: Năm xây dựng hoặc năm của căn hộ.

có 11 trường và 23764 mẫu

          floor         price      rooms            sq        year
Min    0.000000  5.000000e+03   1.000000  8.800000e+00    70.00000
Max   10.000000  1.500000e+07  10.000000  1.007185e+06  2980.00000
Mean   2.808744  6.493536e+05   2.620771  1.027249e+02  2000.55117

a. Số ngôi nhà có số phòng trên 5 từ năm 2000 là: 110
b. Số ngôi nhà từ năm 2000 đến nay có giá trên 500.000 là: 9290
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
data = pd.read_csv(r"D:\Chuyendetientien-Kiemtra\Bai14\Houses.csv", encoding='iso-8859-2')

#Bai 2
# Lọc các cột cần tính toán
cols_to_analyze = ['floor', 'price', 'rooms', 'sq', 'year' ]

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
# a. Số ngôi nhà có số phòng (rooms) trên 5 từ năm 2000
num_houses_rooms_above_5_from_2000 = len(data[(data['rooms'] > 5) & (data['year'] >= 2000)])

# b. Số ngôi nhà từ năm 2000 đến nay có giá trên 500.000
num_houses_price_above_500k_from_2000 = len(data[(data['price'] > 500000) & (data['year'] >= 2000)])

print("a. Số ngôi nhà có số phòng trên 5 từ năm 2000 là:", num_houses_rooms_above_5_from_2000)
print("b. Số ngôi nhà từ năm 2000 đến nay có giá trên 500.000 là:", num_houses_price_above_500k_from_2000)


writeFile(r"D:\Chuyendetientien-Kiemtra\Bai14\ketqua14.txt", result)

# Bài 4
# Tạo lưới 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Danh sách các màu sẽ sử dụng cho từng biểu đồ
colors = ['r', 'g', 'b', 'm']  # Màu đỏ, xanh lá, xanh dương, tím

# a. Biểu đồ price với rooms
axs[0, 0].scatter(data['price'], data['rooms'], color=colors[0], alpha=0.5)
axs[0, 0].set_title('Price vs Rooms')
axs[0, 0].set_xlabel('Price')
axs[0, 0].set_ylabel('Rooms')

# b. Biểu đồ price với sq
axs[0, 1].scatter(data['price'], data['sq'], color=colors[1], alpha=0.5)
axs[0, 1].set_title('Price vs Sq')
axs[0, 1].set_xlabel('Price')
axs[0, 1].set_ylabel('Sq')

# c. Biểu đồ price với floor
axs[1, 0].scatter(data['price'], data['floor'], color=colors[2], alpha=0.5)
axs[1, 0].set_title('Price vs Floor')
axs[1, 0].set_xlabel('Price')
axs[1, 0].set_ylabel('Floor')

# d. Biểu đồ sq với rooms
axs[1, 1].scatter(data['sq'], data['rooms'], color=colors[3], alpha=0.5)
axs[1, 1].set_title('Sq vs Rooms')
axs[1, 1].set_xlabel('Sq')
axs[1, 1].set_ylabel('Rooms')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
