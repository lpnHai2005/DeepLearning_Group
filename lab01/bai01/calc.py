while True:
    try:
        a = float(input('Nhap a: '))
        break
    except ValueError:
        print('Nhap sai! Vui long nhap mot so.')

while True:
    try:
        b = float(input('Nhap b: '))
        break
    except ValueError:
        print('Nhap sai! Vui long nhap mot so.')

# Tính toán
cong = a + b
tru = a - b
nhan = a * b

if b != 0:
    chia = a / b
else:
    chia = 'Khong the chia cho 0'

# Xuất dữ liệu
print(f'{a} + {b} = {cong}')
print(f'{a} - {b} = {tru}')
print(f'{a} * {b} = {nhan}')
print(f'{a} / {b} = {chia}')