import multiprocessing
import os

import hydra
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig


@hydra.main(config_path='../conf', config_name="eval_segmentation")
def run_app(cfg: DictConfig) -> None:
    if cfg.type == 'npy':
        assert cfg.t is not None or cfg.curve
    df = pd.read_csv(cfg.list, names=['filename'])
    df['filename'] = df['filename'].apply(lambda x: x.split('.')[0].split('/')[-1])
    name_list = df['filename'].values
    if not cfg.curve:
        loglist = do_python_eval(cfg.predict_dir, cfg.gt_dir, name_list, 21, cfg.type, cfg.t, printlog=True)
        writelog(cfg.logfile, loglist, cfg.comment)
    else:
        l = []
        for i in range(60):
            t = i / 100.0
            loglist = do_python_eval(cfg.predict_dir, cfg.gt_dir, name_list, 21, cfg.type, t)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%' % (i, t, loglist['mIoU']))
        writelog(cfg.logfile, {'mIoU': l}, cfg.comment)


categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def compare(name_list, predict_folder, gt_folder, num_cls, start, step, TP, P, T, input_type, threshold):
    for idx in range(start, len(name_list), step):
        name = name_list[idx]
        if input_type == 'png':
            predict_file = os.path.join(predict_folder, '%s.png' % name)
            predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder, '%s.npy' % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in predict_dict.keys():
                tensor[key + 1] = predict_dict[key]
            tensor[0, :, :] = threshold
            predict = np.argmax(tensor, axis=0).astype(np.uint8)

        gt_file = os.path.join(gt_folder, '%s.png' % name)
        gt = np.array(Image.open(gt_file))
        cal = gt < 255
        mask = (predict == gt) * cal

        for i in range(num_cls):
            P[i].acquire()
            P[i].value += np.sum((predict == i) * cal)
            P[i].release()
            T[i].acquire()
            T[i].value += np.sum((gt == i) * cal)
            T[i].release()
            TP[i].acquire()
            TP[i].value += np.sum((gt == i) * mask)
            TP[i].release()


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(
        name_list, predict_folder, gt_folder, num_cls, i, 8, TP, P, T, input_type, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == "__main__":
    run_app()
