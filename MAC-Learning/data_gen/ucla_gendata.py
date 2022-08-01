import argparse
import pickle
from tqdm import tqdm
import sys
import random
import cv2

sys.path.extend(['./'])
from data_gen.preprocess import pre_normalization

max_body = 1
num_class = 10
num_joint = 20
max_frame = 300

import numpy as np
import os


def read_skeleton_filter(path):
    numframe = 0
    skeleton_sequence = {}
    skeleton_sequence['frameInfo'] = []
    for file in os.listdir(path):
        numframe += 1
        frame_info = {}
        frame_info['numBody'] = 1
        frame_info['bodyInfo'] = []
        with open(os.path.join(path, file), 'r') as f:
            #print(path, file)
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info['numJoint'] = num_joint
                body_info['jointInfo'] = []
                _ = f.readline()
                for v in range(body_info['numJoint']):
                    joint_info_key = ['x', 'y', 'z', 'unknown']
                    ff=f.readline().split(',')
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, ff)
                    }
                    #print(ff)
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)
    skeleton_sequence['numFrame'] = numframe
        
    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(path):  # 取了前两个body
    
    seq_info = read_skeleton_filter(path)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    '''energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]'''
    
    data = data.transpose(3, 1, 2, 0)  # to (C,T,V,M)
    return data

#semi
def get_semi_data(num_flat, gen_list):
    if (num_flat-1) in gen_list:
        return True
    else:
        return False
#

def get_length(data):
    length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
    return length

def real_resize(data_numpy, length, crop_size):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :length, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)

def gendata(data_path, out_path, phase='eval', semi=5):
    
    sample_name = []
    sample_label = []
    sample_paths = []
    num_train = 0
    num_eval = 0
    gen_list = []
    file_list = []
    class_num_list = []
    for i in range(num_class):
        class_num_list.append(0)
    if phase == 'train':
        for folder in data_path['train']:
            for filelist in os.listdir(folder):
                file_list.append((folder, filelist))
        gen_list = random.sample(range(96),semi)
    else:
        for folder in data_path['eval']:
            for filelist in os.listdir(folder):
                file_list.append((folder, filelist))
    
    for folder, filelist in sorted(file_list):
        action_class = int(filelist[1:3])
        if action_class == 8 or action_class == 9:
            action_class -= 1
        elif action_class == 11 or action_class == 12:
            action_class -= 2
        path = os.path.join(folder, filelist)
        #print(path)
        
        class_num_list[action_class - 1] += 1
    
        if phase == 'eval':
            sample_paths.append(path)
            sample_label.append(action_class - 1)  # to 0-indexed
        else:
            if get_semi_data(class_num_list[action_class - 1], gen_list):
                sample_paths.append(path)
                sample_label.append(action_class - 1)  # to 0-indexed
            else:
                sample_paths.append(path)
                sample_label.append(-1)  # to 0-indexed
    
    # Save labels
    with open('{}/{}_label.pkl'.format(out_path, phase), 'wb') as f:
        pickle.dump((sample_paths, list(sample_label)), f)
    
    # Create data tensor (N,C,T,V,M)
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body), dtype=np.float32)
    
    # resize data to 50 frames
    resize_frame = 50
    fpp = np.zeros((len(sample_label), 3, resize_frame, num_joint, max_body), dtype=np.float32)
    
    # Fill (C,T,V,M) to data tensor (N,C,T,V,M)
    for i, s in enumerate(sample_paths):
        data = read_xyz(s)
        fp[i, :, 0:data.shape[1], :, :] = data
        
        # resize data to 50 frames
        length = get_length(fp[i])
        #print(length)
        if length == 0:
            fpp[i] = fp[i,:,:resize_frame,:,:]
        else:
            fpp[i] = real_resize(fp[i], length, resize_frame)
    
    # Perform preprocessing on data tensor
    fpp = pre_normalization(fpp)
    print(np.shape(fpp)) 
    
    # Save input data (train/eval)
    np.save('{}/{}_data_joint.npy'.format(out_path, phase), fpp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NW-UCLA Data Converter.')
    parser.add_argument('--train_data_path1', default='../nw_ucla/multiview_action/view_1/')
    parser.add_argument('--train_data_path2', default='../nw_ucla/multiview_action/view_2/')
    parser.add_argument('--test_data_path', default='../nw_ucla/multiview_action/view_3/')
    parser.add_argument('--out_folder', default='../data_ucla_acc5_01/')

    #benchmark = ['xsub', 'xview']
    #benchmark = ['acc']
    phase = ['train', 'eval']
    arg = parser.parse_args()

    path_list = {'train':[arg.train_data_path1, arg.train_data_path2], 'eval':[arg.test_data_path]}

    for p in phase:
        out_path = os.path.join(arg.out_folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(p)
        gendata(
            path_list,
            out_path,
            phase=p)
