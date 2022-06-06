# Generates a txt file of test cases for hotel recognition used during training of ArcFace model

import os
import random

data_file_path = "/scratch/guchoadeassis/groupw/preprocessed_mlip_train_data"
test_file_path = "/scratch/guchoadeassis/groupw/arcface-pytorch/test_pair.txt"
dirs = os.listdir(data_file_path)

num_test_lines = 500
len_dirs = len(dirs)
random_range = range(len_dirs-1)

with open(test_file_path, 'w') as test_file:
    # Different
    for line in range(num_test_lines):
        num1, num2 = random.sample(random_range, 2)
        print(len(dirs), num1, num2)
        path1 = os.listdir(os.path.join(data_file_path, dirs[num1]))
        path2 = os.listdir(os.path.join(data_file_path, dirs[num2]))
        file1 = path1[0]
        file2 = path2[0]
        text = f'{dirs[num1]}/{file1} {dirs[num2]}/{file2} 0\n'
        test_file.write(text)
    # Same
    for line in range(num_test_lines):
        num = random.sample(random_range, 1)[0]
        path = os.listdir(os.path.join(data_file_path, dirs[num]))
        if len(path) >= 2:
            file1 = path[0]
            file2 = path[1]
            text = f'{dirs[num]}/{file1} {dirs[num]}/{file2} 1\n'
            test_file.write(text)
        else:
            num_test_lines += 1
    

