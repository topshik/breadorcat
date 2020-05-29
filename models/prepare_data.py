import os
from pathlib import Path
from random import shuffle, seed
import shutil

import pandas as pd

seed(0xBADBEEF)

# copy bread from iFood-2019
ifood_train_labels = pd.read_csv('./source_datasets/ifood-2019-fgvc6/train_labels.csv', index_col='img_name')
ifood_class_list = pd.read_csv('./source_datasets/ifood-2019-fgvc6/class_list.txt', sep=' ', names=['idx', 'name'])
ifood_train_labels['class_name'] = ifood_train_labels['label'].map(
    dict(zip(ifood_class_list.idx, ifood_class_list.name)))

train_part = 0.9
if not os.listdir('./data/train/bread') and not os.listdir('./data/test/bread'):
    src_path = './source_datasets/ifood-2019-fgvc6/train_set'
    files_list = os.listdir(src_path)
    shuffle(files_list)
    vc = ifood_train_labels['class_name'].value_counts()
    thrsh = int(train_part * (vc.gingerbread + vc.garlic_bread))
    cnt = 0
    for i, pic_path in enumerate(files_list):
        if ifood_train_labels.loc[pic_path].class_name in ['garlic_bread', 'gingerbread'] and cnt < thrsh:
            shutil.copy2(src_path + '/' + pic_path, './data/train/bread')
            cnt += 1
            if cnt == thrsh:
                print(f'Collected {cnt} bread pics from iFood successfully (training set)')
        if ifood_train_labels.loc[pic_path].class_name in ['garlic_bread', 'gingerbread'] and thrsh <= cnt:
            shutil.copy2(src_path + '/' + pic_path, './data/test/bread')
            cnt += 1
    print(f'Collected {cnt - thrsh} bread pics from iFood successfully (test set)')
else:
    print('One of the training or test bread sets is already formed')

# collect bread from food11
if not Path('./data/train/bread/0_0.jpg').is_file():
    train_src_path = './source_datasets/food11/food11re/training/0'
    test_src_path = './source_datasets/food11/food11re/validation/0'
    train_cnt, test_cnt = 0, 0
    for i, pic_path in enumerate(os.listdir(train_src_path)):
        shutil.copy2(train_src_path + '/' + pic_path, './data/train/bread')
        train_cnt += 1
    print(f'Collected {train_cnt} bread pics from food11 successfully (training set)')
    for i, pic_path in enumerate(os.listdir(test_src_path)):
        shutil.copy2(test_src_path + '/' + pic_path, './data/test/bread')
        test_cnt += 1
    print(f'Collected {test_cnt} bread pics from food11 successfully (test set)')
else:
    print('Bread pics from Food11 are already collected')

bread_train_size = len(os.listdir('./data/train/bread'))
bread_test_size = len(os.listdir('./data/test/bread'))

# copy cats from cats vs dogs
if not os.listdir('./data/train/cats') and not os.listdir('./data/test/cats'):
    src_path = './source_datasets/dogs-vs-cats/train'
    files_list = os.listdir(src_path)
    shuffle(files_list)
    train_cnt = 0
    test_cnt = 0
    for i, pic_path in enumerate(files_list):
        if pic_path[:3] == 'cat' and train_cnt < bread_train_size:
            shutil.copy2(src_path + '/' + pic_path, './data/train/cats')
            train_cnt += 1
            if train_cnt == bread_train_size:
                print(f'Collected {train_cnt} cat pics from Cats-vs-dogs successfully (training set)')
                train_cnt += 1
        if pic_path[:3] == 'cat' and bread_train_size < train_cnt and test_cnt < bread_test_size:
            shutil.copy2(src_path + '/' + pic_path, './data/test/cats')
            test_cnt += 1
            if test_cnt == bread_test_size:
                print(f'Collected {test_cnt} cat pics from Cats-vs-dogs successfully (test set)')
                break
else:
    print('One of the training or test cats sets is already formed')

# form other images class
if not os.listdir('./data/train/other') and not os.listdir('./data/test/other'):
    src_path = './source_datasets/tiny_imagenet'
    num_folders = len(os.listdir(src_path))
    train_pic_num = 0
    test_pic_num = 0
    for class_folder in os.listdir(src_path):
        for i, pic_path in enumerate(os.listdir(src_path + '/' + class_folder)):
            if i < bread_train_size // num_folders:
                shutil.copy2(src_path + '/' + class_folder + '/' + pic_path,
                             './data/train/other/' + str(train_pic_num) + '.jpg')
                train_pic_num += 1
            elif i - bread_train_size // num_folders < bread_test_size // num_folders:
                shutil.copy2(src_path + '/' + class_folder + '/' + pic_path,
                             './data/test/other/' + str(test_pic_num) + '.jpg')
                test_pic_num += 1
    print(f"Collected {len(os.listdir('./data/train/other'))} pics from tiny ImageNet successfully (training set)")
    print(f"Collected {len(os.listdir('./data/test/other'))} pics from tiny ImageNet successfully (test set)")
else:
    print('One of the training or test other images sets is already formed')

print('\nTraining set')
print(f"Total cats for training: {len(os.listdir('./data/train/cats'))}")
print(f"Total bread for training: {len(os.listdir('./data/train/bread'))}")
print(f"Total other images for training: {len(os.listdir('./data/train/other'))}")
print('Test set')
print(f"Total cats for testing: {len(os.listdir('./data/test/cats'))}")
print(f"Total bread for testing: {len(os.listdir('./data/test/bread'))}")
print(f"Total other images for testing: {len(os.listdir('./data/test/other'))}")

