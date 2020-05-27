import os
import json

solo_videos = './MUSIC_label/MUSIC_solo_videos.json'
solo_videos = json.load(open(solo_videos, 'r'))
solo_videos = solo_videos['videos']

trains = []
vals = []

for _, item in solo_videos.items():
    for i, vid in enumerate(item):
        if i < 5:
            vals.append(vid)
        else:
            trains.append(vid)

videos = open('./data_indicator/music/solo/solo_pairs.txt', 'r')
train_file = open('./data_indicator/music/solo/solo_training.txt', 'w')
val_file = open('./data_indicator/music/solo/solo_validation.txt', 'w')
while True:
    pair = videos.readline()
    if len(pair) == 0:
        break
    vid = pair.split(' ')[0][:-12]
    if vid in trains:
        train_file.write(pair)
    elif vid in vals:
        val_file.write(pair)

videos.close()
train_file.close()
val_file.close()