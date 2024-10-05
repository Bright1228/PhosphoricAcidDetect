def calculate_ratio(file_path):
    zero_count = 0
    one_count = 0
    lines = []

    # 读取文件并存储每组数据的第三行
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())  # 存储每行数据
            if len(lines) == 3:  # 每三行作为一个组
                third_line = lines.pop()  # 取出第三行
                zero_count += third_line.count('0')  # 统计0的数量
                one_count += third_line.count('1')  # 统计1的数量
                lines.clear()  # 清空列表，为下一组数据做准备

    # 计算0和1的总数以及比例
    total_count = zero_count + one_count
    if total_count > 0:
        ratio = total_count / one_count
    else:
        ratio = 0

    return zero_count, one_count, ratio

# 调用函数并传入文件路径
file_path = 'data/data500_35918.fasta'  # 这里替换成你的文件路径
zero_count, one_count, ratio = calculate_ratio(file_path)

# 打印结果
print(f"0的总数: {zero_count}")
print(f"1的总数: {one_count}")
print(f"0和1的比例: {ratio:.2f}")  # 保留两位小数