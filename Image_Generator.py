import cv2
import os, random
import numpy as np
from parameter import letters

# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=9):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.cur_index = 0
        self.last_load_idx = 0

    def next_sample(self):
        batch_img = []
        batch_label = []
        for n in range(self.batch_size):
            idx = self.last_load_idx + n
            if idx >= self.n:
                self.last_load_idx = -n
                idx = 0
                random.shuffle(self.img_dir)
            img = cv2.imread(self.img_dirpath + self.img_dir[idx], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0
            batch_img.append(img)
            batch_label.append(self.img_dir[idx][0:-4])
        self.last_load_idx += self.batch_size
        return batch_img, batch_label

    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)
            img, text = self.next_sample()
            for i in range(self.batch_size):
                img[i] = img[i].T
                img[i] = np.expand_dims(img[i], -1)
                X_data[i] = img[i]
                Y_data[i] = text_to_labels(text[i])
                label_length[i] = len(text[i])
            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
