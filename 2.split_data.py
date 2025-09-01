
import csv
import random

# --- 配置 ---
INPUT_FILE = './data/finetuning_data.csv'
TRAIN_FILE = './data/train.csv'
TEST_FILE = './data/test.csv'
TRAIN_RATIO = 0.8  # 80% 的数据用于训练，其余用于测试

# --- 脚本开始 ---

def split_data():
    """读取CSV文件，打乱顺序，并按比例划分为训练集和测试集。"""
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 读取表头
            data = [row for row in reader] # 读取所有数据行
    except FileNotFoundError:
        print(f"错误：输入文件 '{INPUT_FILE}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return

    # 打乱数据顺序
    random.shuffle(data)

    # 计算切分点
    split_index = int(len(data) * TRAIN_RATIO)

    # 切分数据
    train_data = data[:split_index]
    test_data = data[split_index:]

    # 写入训练集文件
    try:
        with open(TRAIN_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(train_data)
        print(f"成功创建训练集文件：'{TRAIN_FILE}' (包含 {len(train_data)} 条数据)")
    except Exception as e:
        print(f"写入训练集文件时发生错误：{e}")

    # 写入测试集文件
    try:
        with open(TEST_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(test_data)
        print(f"成功创建测试集文件：'{TEST_FILE}' (包含 {len(test_data)} 条数据)")
    except Exception as e:
        print(f"写入测试集文件时发生错误：{e}")

if __name__ == '__main__':
    split_data()
