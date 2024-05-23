'''
    ID: Mã số định danh của khách hàng.
    Age: Tuổi của khách hàng (đã tròn năm).
    Experience: Số năm kinh nghiệm làm việc chuyên nghiệp.
    Income: Thu nhập hàng năm của khách hàng (nghìn đô la).
    ZIP Code: Mã bưu chính của địa chỉ nhà khách hàng.
    Family: Quy mô gia đình của khách hàng (số thành viên).
    CCAvg: Chi tiêu trung bình hàng tháng cho thẻ tín dụng (nghìn đô la).
    Education: Trình độ học vấn.1: Đại học;2: Sau đại học;3: Cao học/Chuyên nghiệp.
    Mortgage: Giá trị thế chấp nhà ở nếu có (nghìn đô la).
    Personal Loan: Khách hàng có chấp nhận khoản vay cá nhân được đề nghị trong chiến dịch gần đây hay không?
    Securities Account: Khách hàng có tài khoản chứng khoán tại ngân hàng hay không?
    CD Account: Khách hàng có tài khoản tiền gửi có kỳ hạn (CD) tại ngân hàng hay không?
    Online: Khách hàng có sử dụng dịch vụ ngân hàng trực tuyến hay không?
    Credit card: Khách hàng có sử dụng thẻ tín dụng do ngân hàng phát hành hay không?

    a. Có 14 trường và 5000 mẫu
    
              Age  Experience    Income
    Min   23.0000     -3.0000    8.0000
    Max   67.0000     43.0000  224.0000
    Mean  45.3384     20.1046   73.7742
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
data = pd.read_csv(r"D:\Chuyendetientien-Kiemtra\Bai1\Bank_Personal_Loan_Modelling001.csv")

#Bai 2
# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': data.min(), 'Max': data.max(), 'Mean': data.mean()}, index=['Age', 'Experience', 'Income'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

#Bai 3
# a. Người độ tuổi dưới 30 mà có thu nhập trên 50.000$
num_a = ((data['Age'] < 30) & (data['Income'] > 50)).sum()
print('Số người độ tuổi dưới 30 mà có thu nhập trên 50.000$:', num_a)

# b. Người có độ tuổi dưới 30 mà có kinh nghiệm trên 5 năm
num_b = ((data['Age'] < 30) & (data['Experience'] > 5)).sum()
print('Số người có độ tuổi dưới 30 mà có kinh nghiệm trên 5 năm:', num_b)

#Bài 4
# Biểu diễn các quan hệ dữ liệu bằng biểu đồ:
import matplotlib.pyplot as plt

# Tạo 4 biểu đồ trên cùng một hình
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# a. Tuổi với kinh nghiệm
axs[0, 0].scatter(data['Age'], data['Experience'], color='red')
axs[0, 0].set_xlabel('Tuổi')
axs[0, 0].set_ylabel('Kinh nghiệm')
axs[0, 0].set_title('Tuổi với kinh nghiệm')

# b. Tuổi với thu nhập
axs[0, 1].scatter(data['Age'], data['Income'], color='green')
axs[0, 1].set_xlabel('Tuổi')
axs[0, 1].set_ylabel('Thu nhập')
axs[0, 1].set_title('Tuổi với thu nhập')

# c. Tuổi với chi tiêu trung bình trong tháng
axs[1, 0].scatter(data['Age'], data['CCAvg'], color='blue')
axs[1, 0].set_xlabel('Tuổi')
axs[1, 0].set_ylabel('Chi tiêu trung bình trong tháng')
axs[1, 0].set_title('Tuổi với chi tiêu trung bình trong tháng')

# d. Thu nhập với chi tiêu trung bình trong tháng
axs[1, 1].scatter(data['Income'], data['CCAvg'], color='purple')
axs[1, 1].set_xlabel('Thu nhập')
axs[1, 1].set_ylabel('Chi tiêu trung bình trong tháng')
axs[1, 1].set_title('Thu nhập với chi tiêu trung bình trong tháng')

# Điều chỉnh khoảng cách giữa các biểu đồ
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()

data = f'''
    1.
        a. Có 14 trường và 5000 mẫu
        b.  ID: Mã số định danh của khách hàng.
            Age: Tuổi của khách hàng (đã tròn năm).
            Experience: Số năm kinh nghiệm làm việc chuyên nghiệp.
            Income: Thu nhập hàng năm của khách hàng (nghìn đô la).
            ZIP Code: Mã bưu chính của địa chỉ nhà khách hàng.
            Family: Quy mô gia đình của khách hàng (số thành viên).
            CCAvg: Chi tiêu trung bình hàng tháng cho thẻ tín dụng (nghìn đô la).
            Education: Trình độ học vấn.1: Đại học;2: Sau đại học;3: Cao học/Chuyên nghiệp.
            Mortgage: Giá trị thế chấp nhà ở nếu có (nghìn đô la).
            Personal Loan: Khách hàng có chấp nhận khoản vay cá nhân được đề nghị trong chiến dịch gần đây hay không?
            Securities Account: Khách hàng có tài khoản chứng khoán tại ngân hàng hay không?
            CD Account: Khách hàng có tài khoản tiền gửi có kỳ hạn (CD) tại ngân hàng hay không?
            Online: Khách hàng có sử dụng dịch vụ ngân hàng trực tuyến hay không?
            Credit card: Khách hàng có sử dụng thẻ tín dụng do ngân hàng phát hành hay không?
    2.
{summary_table_transposed}
    3.
        a. Số người độ tuổi dưới 30 mà có thu nhập trên 50.000$: {num_a}
        b. Số người có độ tuổi dưới 30 mà có kinh nghiệm trên 5 năm: {num_b}
          '''

# Các kết quả ở mục 1, 2, 3 được ghi vào file ketqua.txt
writeFile(r"D:\Chuyendetientien-Kiemtra\Bai1\ketqu01.txt", data)
