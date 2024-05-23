result = '''

date: Ngày tháng theo định dạng dd/mm/yyyy. Đây là ngày lấy mẫu dữ liệu.

time: Thời gian theo định dạng hh:mm:ss. Đây là thời điểm trong ngày mà dữ liệu được lấy mẫu.

global_active_power: Công suất hoạt động toàn cầu trung bình phút của hộ gia đình (tính bằng kilowatt). Đây là công suất điện mà hộ gia đình tiêu thụ trong một phút.

global_reactive_power: Công suất phản kháng toàn cầu trung bình phút của hộ gia đình (tính bằng kilowatt). Đây là công suất phản kháng mà hộ gia đình tiêu thụ, liên quan đến năng lượng bị tích trữ trong các thiết bị như cuộn cảm và tụ điện.

voltage: Điện áp trung bình phút (tính bằng volt). Đây là mức điện áp trung bình trong hộ gia đình trong một phút.

global_intensity: Cường độ dòng điện toàn cầu trung bình phút của hộ gia đình (tính bằng ampere). Đây là cường độ dòng điện trung bình mà hộ gia đình sử dụng trong một phút.

sub_metering_1: Năng lượng tiêu thụ của thiết bị đo lường số 1 (tính bằng watt-giờ năng lượng hoạt động). Điều này tương ứng với khu vực nhà bếp, bao gồm chủ yếu máy rửa chén, lò nướng và lò vi sóng (bếp không sử dụng điện mà sử dụng gas).

sub_metering_2: Năng lượng tiêu thụ của thiết bị đo lường số 2 (tính bằng watt-giờ năng lượng hoạt động). Điều này tương ứng với phòng giặt, bao gồm máy giặt, máy sấy quần áo, tủ lạnh và đèn.

sub_metering_3: Năng lượng tiêu thụ của thiết bị đo lường số 3 (tính bằng watt-giờ năng lượng hoạt động). Điều này tương ứng với máy nước nóng điện và điều hòa không khí.

có 9 trường và  2075259  mẫu

      Global_active_power  Global_reactive_power  Global_intensity  Sub_metering_1  Sub_metering_2  Sub_metering_3
Min              0.076000               0.000000          0.200000        0.000000         0.00000        0.000000
Max             11.122000               1.390000         48.400000       88.000000        80.00000       31.000000
Mean             1.091615               0.123714          4.627759        1.121923         1.29852        6.458447

Tổng số năng lượng tiêu thụ (Global_active_power) trong ngày 16/12/2006: 1209.176 kW
Tổng số năng lượng tiêu thụ (Global_reactive_power) trong ngày 16/12/2006: 34.922 kW
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def writeFile(filepath, data):
    with open(file=filepath, mode='w+', encoding='utf-8', errors='ignore') as f:
        f.write(data)

# Đọc dữ liệu từ file txt
file_path = r"D:\Chuyendetientien-Kiemtra\Bai15\household_power_consumption.txt"
data = pd.read_csv(file_path, sep=';', parse_dates=['Date'], dayfirst=True, low_memory=False)

# Chuyển đổi các cột liên quan thành kiểu float để đảm bảo có thể vẽ biểu đồ
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce')
data['Global_intensity'] = pd.to_numeric(data['Global_intensity'], errors='coerce')
data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'], errors='coerce')
data['Sub_metering_2'] = pd.to_numeric(data['Sub_metering_2'], errors='coerce')
data['Sub_metering_3'] = pd.to_numeric(data['Sub_metering_3'], errors='coerce')

#Bai 2
# Lọc các cột cần tính toán
cols_to_analyze = ['Global_active_power', 'Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

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
# Lọc dữ liệu cho ngày 16/12/2006
filtered_data = data[data['Date'] == '2006-12-16']

# Chuyển đổi các cột 'Global_active_power' và 'Global_reactive_power' thành kiểu float
filtered_data['Global_active_power'] = pd.to_numeric(filtered_data['Global_active_power'], errors='coerce')
filtered_data['Global_reactive_power'] = pd.to_numeric(filtered_data['Global_reactive_power'], errors='coerce')

# Tính tổng số năng lượng
total_active_power = filtered_data['Global_active_power'].sum()
total_reactive_power = filtered_data['Global_reactive_power'].sum()

print(f"Tổng số năng lượng tiêu thụ (Global_active_power) trong ngày 16/12/2006: {total_active_power} kW")
print(f"Tổng số năng lượng tiêu thụ (Global_reactive_power) trong ngày 16/12/2006: {total_reactive_power} kW")

writeFile(r"D:\Chuyendetientien-Kiemtra\Bai15\ketqua15.txt", result)

# Bài 4


# Tạo lưới 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Danh sách các màu sẽ sử dụng cho từng biểu đồ
colors = ['r', 'g', 'b', 'm']  # Màu đỏ, xanh lá, xanh dương, tím

# Biểu đồ 1: Global_intensity với Sub_metering_1
axs[0, 0].scatter(data['Global_intensity'], data['Sub_metering_1'], color=colors[0])
axs[0, 0].set_title('Global Intensity vs Sub_metering_1')
axs[0, 0].set_xlabel('Global Intensity (A)')
axs[0, 0].set_ylabel('Sub_metering_1 (Wh)')

# Biểu đồ 2: Global_intensity với Sub_metering_2
axs[0, 1].scatter(data['Global_intensity'], data['Sub_metering_2'], color=colors[1])
axs[0, 1].set_title('Global Intensity vs Sub_metering_2')
axs[0, 1].set_xlabel('Global Intensity (A)')
axs[0, 1].set_ylabel('Sub_metering_2 (Wh)')

# Biểu đồ 3: Global_intensity với Sub_metering_3
axs[1, 0].scatter(data['Global_intensity'], data['Sub_metering_3'], color=colors[2])
axs[1, 0].set_title('Global Intensity vs Sub_metering_3')
axs[1, 0].set_xlabel('Global Intensity (A)')
axs[1, 0].set_ylabel('Sub_metering_3 (Wh)')

# Biểu đồ 4: Sub_metering_3 với Sub_metering_2
axs[1, 1].scatter(data['Sub_metering_3'], data['Sub_metering_2'], color=colors[3])
axs[1, 1].set_title('Sub_metering_3 vs Sub_metering_2')
axs[1, 1].set_xlabel('Sub_metering_3 (Wh)')
axs[1, 1].set_ylabel('Sub_metering_2 (Wh)')

# Tinh chỉnh khoảng cách giữa các subplot
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()


