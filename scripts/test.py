import os
import sys

# # 获取当前脚本所在目录的父目录的父目录
# grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # 将父目录添加到 sys.path
# sys.path.append(grandparent_dir)

for i in sys.path:
    print(i)

