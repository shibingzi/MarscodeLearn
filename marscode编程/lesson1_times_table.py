# 定义乘法表的大小
size = 10

# 生成乘法表
for i in range(1, size + 1):
    for j in range(1, size + 1):
        print(f"{i} x {j} = {i*j}", end="\t")
    print()  # 换行
