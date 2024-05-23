result = '''
1. age: Tuổi của bệnh nhân (đơn vị: năm).
2. anaemia: Giảm số lượng tế bào máu đỏ hoặc huyết quản (hemoglobin) (boolean).
3. high blood pressure: Xác định xem bệnh nhân có bị cao huyết áp hay không (boolean).
4. creatinine phosphokinase (CPK): Mức độ của enzyme CPK trong máu (đơn vị: mcg/L).
5. diabetes: Xác định xem bệnh nhân có bị tiểu đường hay không (boolean).
6. ejection fraction: Tỷ lệ phần trăm máu rời khỏi tim ở mỗi lần co bóp (phần trăm).
7. platelets: Số lượng tiểu cầu trong máu (đơn vị: kiloplatelets/mL).
8. sex: Giới tính của bệnh nhân (nữ hoặc nam) (binary).
9. serum creatinine: Mức độ creatinine trong huyết thanh (đơn vị: mg/dL).
10. serum sodium: Mức độ natri trong huyết thanh (đơn vị: mEq/L).
11. smoking: Xác định xem bệnh nhân có hút thuốc lá hay không (boolean).
12. time: Thời gian theo dõi (đơn vị: ngày).
13. death event: [target] Xác định xem bệnh nhân có mất tích trong thời gian theo dõi hay không (boolean).

có 13 trường và 299 mẫu

            age  creatinine_phosphokinase  ejection_fraction  high_blood_pressure      platelets
Min   40.000000                 23.000000          14.000000             0.000000   25100.000000
Max   95.000000               7861.000000          80.000000             1.000000  850000.000000
Mean  60.833893                581.839465          38.083612             0.351171  263358.029264

a. Số bệnh nhân tuổi trên 60 có creatinine_phosphokinase dưới 300 là: 79
b. Số bệnh nhân nữ tuổi trên 60 là: 0
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
data = pd.read_csv(r"D:\Chuyendetientien-Kiemtra\Bai13\heart_failure_clinical_records_dataset.csv")

#Bai 2
# Lọc các cột cần tính toán
cols_to_analyze = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'high_blood_pressure', 'platelets' ]

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
df = pd.DataFrame(data)
# Tính số bệnh nhân tuổi trên 60 có creatinine_phosphokinase dưới 300
num_patients_cpk_under_300_over_60 = len(df[(df['age'] > 60) & (df['creatinine_phosphokinase'] < 300)])

# Tính số bệnh nhân nữ tuổi trên 60
num_female_patients_over_60 = len(df[(df['age'] > 60) & (df['sex'] == 'female')])

# In kết quả
print("a. Số bệnh nhân tuổi trên 60 có creatinine_phosphokinase dưới 300 là:", num_patients_cpk_under_300_over_60)
print("b. Số bệnh nhân nữ tuổi trên 60 là:", num_female_patients_over_60)

writeFile(r"D:\Chuyendetientien-Kiemtra\Bai13\ketqua13.txt", result)

# Bài 4

# Tạo lưới 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Danh sách các màu sẽ sử dụng cho từng biểu đồ
colors = ['r', 'g', 'b', 'm']  # Màu đỏ, xanh lá, xanh dương, tím

# Biểu đồ a: age với creatinine_phosphokinase
axs[0, 0].scatter(df['age'], df['creatinine_phosphokinase'], color=colors[0])
axs[0, 0].set_title('Age vs Creatinine Phosphokinase')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Creatinine Phosphokinase')

# Biểu đồ b: age với ejection_fraction
axs[0, 1].scatter(df['age'], df['ejection_fraction'], color=colors[1])
axs[0, 1].set_title('Age vs Ejection Fraction')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Ejection Fraction')

# Biểu đồ c: age với high_blood_pressure
axs[1, 0].scatter(df['age'], df['high_blood_pressure'], color=colors[2])
axs[1, 0].set_title('Age vs High Blood Pressure')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('High Blood Pressure')

# Biểu đồ d: high_blood_pressure với ejection_fraction
axs[1, 1].scatter(df['high_blood_pressure'], df['ejection_fraction'], color=colors[3])
axs[1, 1].set_title('High Blood Pressure vs Ejection Fraction')
axs[1, 1].set_xlabel('High Blood Pressure')
axs[1, 1].set_ylabel('Ejection Fraction')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
