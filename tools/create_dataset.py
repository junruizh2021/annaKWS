import os
import glob
import linecache
import numpy as np
from tqdm import tqdm

"""
1. 使用himia数据集作为唤醒词（正例）数据来源，随机选出X段音频，转为16k采样
2. 采用AISHELL2数据集作为非唤醒词（反例）数据来源，随机选出X段音频，转为16k采样
3. 创建文件列表wav.scp文件，形如：
    first.wav /path/to/first.wav 
    second.wav /path/to/second.wav
4. 创建标签列表文件，形如：
    first.wav 0
    second.wav -1
5. use the script in wav_to_duration.sh to get the "wav.dur" file, and then run tools/make_list.py
    bash tools/wav_to_duration.sh /path/to/wav.scp /path/to/wav.dur
    python tools/make_list.py /path/to/wav.scp /path/to/label_text /path/to/wav.dur /path/to/data.list
"""
# himia数据集路径 作为正例
HIMIA_DATASET_PATH = 'E:/HiMIA/train/SPEECHDATA'
# AISHELL2数据集路径 作为反例
AISHELL2_DATASET_PATH = 'E:/AISHELL2'
# 训练数据输出位置
training_data_dir = 'E:/kws-dataset/train'
training_data_p_dist_dir = 'E:/kws-dataset/train/positive'
training_data_n_dist_dir = 'E:/kws-dataset/train/negative'
# 验证数据输出位置
val_data_dir = 'E:/kws-dataset/val'
val_data_dist_p_dir = 'E:/kws-dataset/val/positive'
val_data_dist_n_dir = 'E:/kws-dataset/val/negative'
# 训练数据：1000个
num_train_files = 1000
num_val_files = 100


def iter_count(file_name):
    # 获取文件的行数
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def get_line_context(file_path, line_number):
    # 获取文件第line_number行的内容
    return linecache.getline(file_path, line_number).strip()


def sample_to_wakeup_words():
    """
    HI-MIA-SLR85 ---> wakeup data
    """
    training_data_list_file = os.path.join(HIMIA_DATASET_PATH, "train.scp")  # 内部是相对路径
    total_samples_num = iter_count(training_data_list_file)

    def sample_and_trans(sample_nums, dst_path):
        training_sample_index = np.random.randint(1, total_samples_num, (sample_nums,))
        for sample_index in training_sample_index:
            wave_filename = get_line_context(training_data_list_file, sample_index)
            src_wave_file_path = os.path.join(HIMIA_DATASET_PATH, wave_filename)
            dst_wave_file_path = os.path.join(dst_path, "p_" + os.path.basename(src_wave_file_path))
            os.popen("ffmpeg -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(src_wave_file_path, dst_wave_file_path))
        pass

    sample_and_trans(num_train_files, training_data_p_dist_dir)
    sample_and_trans(num_val_files, val_data_dist_p_dir)


def sample_to_not_wakeup_words():
    val_data_list_file = os.path.join(AISHELL2_DATASET_PATH, "all.txt")  # 内部是绝对路径
    total_samples_num = iter_count(val_data_list_file)

    def sample_and_trans(sample_nums, dst_path):
        training_sample_index = np.random.randint(1, total_samples_num, (sample_nums,))
        for sample_index in training_sample_index:
            src_wave_file_path = get_line_context(val_data_list_file, sample_index)
            # src_wave_file_path = os.path.join(HIMIA_DATASET_PATH, wave_filename)
            dst_wave_file_path = os.path.join(dst_path, "n_" + os.path.basename(src_wave_file_path))
            os.popen("ffmpeg -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(src_wave_file_path, dst_wave_file_path))
        pass

    sample_and_trans(num_train_files, training_data_n_dist_dir)
    sample_and_trans(num_val_files, val_data_dist_n_dir)
    pass


def create_scp_file_and_label(src):
    file_list = glob.glob('{}/*/*.wav'.format(src), recursive=True)

    with open(os.path.join(src, 'wav.scp'), 'w') as f:
        for file_name in tqdm(file_list, desc='Creating wav.scp file'):
            f.write("{} {}\n".format(os.path.basename(file_name), file_name.replace('\\', '/')))

    with open(os.path.join(src, 'label.txt'), 'w') as f:
        for file_name in tqdm(file_list, desc='Creating label.scp file'):
            f.write("{} {}\n".format(os.path.basename(file_name), (0 if file_name.find('positive') != -1 else -1)))
    pass


if __name__ == '__main__':
    sample_to_wakeup_words()
    sample_to_not_wakeup_words()

    create_scp_file_and_label(training_data_dir)
    create_scp_file_and_label(val_data_dir)
