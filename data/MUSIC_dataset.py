import numpy as np
import librosa
from PIL import Image, ImageEnhance
import pickle
import random
import os
import torchvision.transforms as transforms
import json
import torch

def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image


class MUSIC_Dataset(object):

    def __init__(self, data_root, data_list_file, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        self.audio_root = os.path.join(data_root, 'audio_frames')
        self.video_root = os.path.join(data_root, 'video_frames')

        with open(data_list_file,'r') as fid:
            pairs = [line.strip().split(' ') for line in fid.readlines()]
       
        self.sample_label= self._parse_csv(self.opt.json_file)

        self.audio_list = []
        self.video_list = []
        self.label_list = []

        for each in pairs:
            audio = each[0]
            video = each[1]
            assert audio[:-5] == video[:-4]
            audio_path = os.path.join(self.audio_root, audio[:-5])
            video_path = os.path.join(self.video_root, video[:-4])

            audio_samples= os.listdir(audio_path)
            for item in range(len(audio_samples)):
                audio_segment = audio_samples[item]
                video_segment = os.path.join(video_path, 'frame_'+audio_segment[:3])
                if os.path.exists(video_segment):
                    self.audio_list.append(os.path.join(audio_path, audio_segment))
                    self.video_list.append(os.path.join(video_path, video_segment))

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]

        self.img_transform = transforms.Compose(img_transform_list)
        #self.audio_transform = audio_transform

    def __len__(self):
        return len(self.audio_list)


    def _parse_csv(self, json_file):
        f = open(json_file, encoding='utf-8')
        content = f.read()
        ins_indicator = json.loads(content)
        ins_indicator = ins_indicator['videos']
        ins_list = [*ins_indicator]
        sample_label = {}
        pickle.dump(ins_list, open('keylist.pkl', 'wb'))
        for i in range(len(ins_list)):
            current_list = ins_indicator[ins_list[i]]
            for j in range(len(current_list)):
                sample_label[current_list[j]] = i
        return sample_label


    def __getitem__(self, index):

        # positive
        cur_audio_segment = self.audio_list[index]
        posi_video_segment = self.video_list[index]
        if self.opt.mode == 'train':
            posi_video_segment_img = random.choice(os.listdir(posi_video_segment))
        else:
            posi_video_segment_img = os.listdir(posi_video_segment)[0]

        # load data
        with open(cur_audio_segment, 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        cur_audio_data = np.expand_dims(cur_audio_data, 0)
        posi_img_path = os.path.join(posi_video_segment, posi_video_segment_img)
        posi_img = Image.open(posi_img_path)
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            posi_img = augment_image(posi_img)
        posi_img = self.img_transform(posi_img)
        posi_label = self.sample_label[posi_video_segment[-28:-17]]
        # TODO: here may need normalization

        # negative
        while(1):
            nega_video_segment = random.choice(self.video_list)
            if nega_video_segment[-28:-17] != posi_video_segment[-28:-17]:
                break
        nega_video_segment_img = random.choice(os.listdir(nega_video_segment))
        nega_img_path = os.path.join(nega_video_segment, nega_video_segment_img)
        nega_img = Image.open(nega_img_path)       
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            nega_img = augment_image(nega_img)
        nega_img = self.img_transform(nega_img)
        nega_label = self.sample_label[nega_video_segment[-28:-17]]

        if self.opt.mode == 'train':
            return cur_audio_data, posi_img, nega_img, posi_label, nega_label, posi_video_segment, cur_audio_segment
        return cur_audio_data, posi_img, nega_img, posi_label, nega_label, posi_img_path, cur_audio_segment

class MUSIC_Dataset_(object):

    def __init__(self, data_root, data_list_file, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        if self.opt.mode == 'train':
            self.audio_root = '/home/yuxi/ruiq/AudioVisual/multiple-sound-source-localization/synthesize/train/audio'
            self.video_root = '/home/yuxi/ruiq/AudioVisual/multiple-sound-source-localization/synthesize/train/video'
        else:
            self.audio_root = '/home/yuxi/ruiq/AudioVisual/multiple-sound-source-localization/synthesize/test/audio'
            self.video_root = '/home/yuxi/ruiq/AudioVisual/multiple-sound-source-localization/synthesize/test/video'
        self.box_root = '/home/yuxi/ruiq/AudioVisual/multiple-sound-source-localization/synthesize/test/box'

        self.audio_list = os.listdir(self.audio_root)
        self.video_list = os.listdir(self.video_root)
        self.box_list = os.listdir(self.box_root)
        self.audio_list.sort()
        self.video_list.sort()
        self.box_list.sort()
        
        assert len(self.audio_list) == len(self.video_list)
       
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]

        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):

        # positive
        cur_audio_segment = self.audio_list[index]
        posi_video_segment = self.video_list[index]
        if self.opt.mode == 'val':
            box_segment = self.box_list[index]
            
        # load data
        with open(os.path.join(self.audio_root, cur_audio_segment), 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        cur_audio_data = np.expand_dims(cur_audio_data, 0)
        posi_img_path = os.path.join(self.video_root, posi_video_segment)
        posi_img = Image.open(posi_img_path)
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            posi_img = augment_image(posi_img)
        posi_img = self.img_transform(posi_img)

        while(1):
            nega_video_segment = random.choice(self.video_list)
            if nega_video_segment != posi_video_segment:
                break
        nega_img_path = os.path.join(self.video_root, nega_video_segment)
        nega_img = Image.open(nega_img_path)       
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            nega_img = augment_image(nega_img)
        nega_img = self.img_transform(nega_img)
        
        if self.opt.mode == 'val':
            box = np.load(os.path.join(self.box_root, box_segment))
            return cur_audio_data, posi_img, nega_img, torch.tensor(0), torch.tensor(0), torch.tensor(0), box
        return cur_audio_data, posi_img, nega_img, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)

class MUSIC_AV_Classify(object):

    def __init__(self, video_dirs, aud_dirs, label, opt):
        self.opt = opt
        self.video_dirs = video_dirs
        self.aud_dirs = aud_dirs
        self.label = label
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor(), 
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.video_dirs)


    def __getitem__(self, index):
        video_segment_img = random.choice(os.listdir(self.video_dirs[index]))
        img_path = os.path.join(self.video_dirs[index], video_segment_img)

        img = Image.open(img_path)
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            img = augment_image(img)
        img_data = self.img_transform(img)


        with open(self.aud_dirs[index], 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        audio_data = np.expand_dims(cur_audio_data, 0)

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            return audio_data, img_data
        else:
            return audio_data, img_data, self.label[index]