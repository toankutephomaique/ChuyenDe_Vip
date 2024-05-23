result = '''
01 age: Tuổi của bệnh nhân.
02 sex: Giới tính của bệnh nhân.(0 = nữ, 1 = nam).
03 cp (chest pain): Loại đau ngực mà bệnh nhân trải qua.
    Định dạng: Số nguyên (1-4).
    1 = Đau thắt ngực kiểu điển hình.
    2 = Đau thắt ngực kiểu không điển hình.
    3 = Đau ngực không do tim.
    4 = Không có triệu chứng đau ngực.
04 trestbps (resting blood pressure): Huyết áp tâm thu lúc nghỉ ngơi (mm Hg).
05 chol (serum cholesterol): Cholesterol trong huyết thanh (mg/dl).
06 fbs (fasting blood sugar): Đường huyết lúc đói (> 120 mg/dl, 1 = đúng, 0 = sai).
07 restecg (resting electrocardiographic results): Kết quả điện tâm đồ lúc nghỉ ngơi.
    Định dạng: Số nguyên (0-2).
    0 = Bình thường.
    1 = Có bất thường ST-T (t-wave inversions và/hoặc ST elevation hoặc depression > 0.05 mV).
    2 = Phì đại thất trái theo tiêu chuẩn Estes.
08 thalach (maximum heart rate achieved): Nhịp tim tối đa đạt được trong quá trình kiểm tra gắng sức.
09 exang (exercise induced angina): Đau thắt ngực do gắng sức (1 = có, 0 = không).
10 oldpeak:  Sự giảm ST (ST depression) được gây ra bởi bài tập so với lúc nghỉ ngơi.
11 slope (slope of the peak exercise ST segment): Độ dốc của đoạn ST cao nhất khi gắng sức.
    Định dạng: Số nguyên (1-3).
    1 = Độ dốc đi lên.
    2 = Phẳng.
    3 = Độ dốc đi xuống.
12 ca (number of major vessels colored by fluoroscopy): Số lượng mạch chính (0-3) được chiếu sáng bằng huỳnh quang.
13 thal: Thalassemia (chỉ tình trạng thiếu máu hoặc bất thường huyết học được xác định thông qua kiểm tra Thallium stress test)
    (3 = bình thường, 
    6 = cố định khiếm khuyết, 
    7 = thalassemia đảo ngược).
14 class: Biến mục tiêu cho biết tình trạng bệnh tim của bệnh nhân 
    (0 = không có bệnh, 1,2,3,4 = có bệnh(theo mức độ nặng dần)).

có 14 trường và 303 mẫu

            age    trestbps        chol     thalach   oldpeak
Min   29.000000   94.000000  126.000000   71.000000  0.000000
Max   77.000000  200.000000  564.000000  202.000000  6.200000
Mean  54.438944  131.689769  246.693069  149.607261  1.039604

a. Số lượng bệnh nhân tuổi trên 60 và có huyết áp trên 160: 4
b. Số lượng bệnh nhân nữ tuổi trên 60: 34
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
data = pd.read_csv(r"D:\Chuyendetientien-Kiemtra\Bai12\Cleveland_hd.csv")

#Bai 2
# Lọc các cột cần tính toán
cols_to_analyze = ['age','trestbps','chol','thalach','oldpeak' ]

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
# a. Số lượng bệnh nhân tuổi trên 60 và có huyết áp (trestbps) trên 160
over_60_high_bp_count = len(data[(data['age'] > 60) & (data['trestbps'] > 160)])
print(f"a. Số lượng bệnh nhân tuổi trên 60 và có huyết áp trên 160: {over_60_high_bp_count}")

# b. Số lượng bệnh nhân nữ tuổi trên 60
female_over_60_count = len(data[(data['sex'] == 0) & (data['age'] > 60)])
print(f"b. Số lượng bệnh nhân nữ tuổi trên 60: {female_over_60_count}")

writeFile(r"D:\Chuyendetientien-Kiemtra\Bai12\ketqua12.txt", result)

# Bài 4


# Tạo lưới 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))
df = pd.DataFrame(data)
# Danh sách các màu sẽ sử dụng cho từng biểu đồ
colors = ['r', 'g', 'b', 'm']  # Màu đỏ, xanh lá, xanh dương, tím
# a. age với trestbps
axs[0, 0].scatter(df['age'], df['trestbps'], color=colors[0])
axs[0, 0].set_title('Age vs Resting Blood Pressure')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Resting Blood Pressure (mm Hg)')

# b. age với chol
axs[0, 1].scatter(df['age'], df['chol'], color=colors[1])
axs[0, 1].set_title('Age vs Serum Cholesterol')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Serum Cholesterol (mg/dl)')

# c. age với thalach
axs[1, 0].scatter(df['age'], df['thalach'], color=colors[2])
axs[1, 0].set_title('Age vs Maximum Heart Rate')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Maximum Heart Rate')

# d. trestbps với chol
axs[1, 1].scatter(df['trestbps'], df['chol'], color=colors[3])
axs[1, 1].set_title('Resting Blood Pressure vs Serum Cholesterol')
axs[1, 1].set_xlabel('Resting Blood Pressure (mm Hg)')
axs[1, 1].set_ylabel('Serum Cholesterol (mg/dl)')

# Điều chỉnh khoảng cách giữa các biểu đồ
plt.tight_layout()
plt.show()