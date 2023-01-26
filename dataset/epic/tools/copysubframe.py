# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import pandas as pd
import csv

import mmcv

data_root = '/data/kide004/repos/VideoMAE/dataset/epic/test'
video_root = '/data/kide004/repos/VideoMAE/dataset/epic/test'
anno_root = f'{data_root}/annotations'
anno_file = '/data/kide004/repos/VideoMAE/dataset/epic/test_aanotation.csv'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = '/data/kide004/repos/VideoMAE/dataset/epic/trim_video'

videos = os.listdir(video_root)
videos = set(videos)
cleaned = pd.read_csv(anno_file, header=None, delimiter=',')
narration_id = list(cleaned[0][:])
video_id = list(cleaned[2][:])
start = list(cleaned[4][:])
stop = list(cleaned[5][:])
verb = list(cleaned[10][:])
noun = list(cleaned[12][:])
event_annotation = []

mmcv.mkdir_or_exist(event_root)

def convert_second(date_time):
     hour = float(date_time[0:2]) * 3600
     min = float(date_time[3:5]) * 60
     sec = float(date_time[6:])
     return hour + min + sec

for i, k in enumerate(video_id):
     if k + '.mp4' not in videos:
          print(f'video {k} has not been downloaded')
          continue

     video_path = osp.join(video_root, k + '.mp4')

     start_stamptime = convert_second(start[i])
     end_stamptime = convert_second(stop[i])
     event_name = narration_id[i]

     output_filename = event_name + '.mp4'

     command = [
          'ffmpeg', '-i',
          '"%s"' % video_path, '-ss',
          str(start_stamptime), '-t',
          str(end_stamptime - start_stamptime), '-c:v', 'libx264', '-c:a', 'copy',
          '-threads', '8', '-loglevel', 'panic',
          '"%s"' % osp.join(event_root, output_filename)
     ]
     command = ' '.join(command)
     try:
          subprocess.check_output(
               command, shell=True, stderr=subprocess.STDOUT)
     except subprocess.CalledProcessError:
          print(
               f'Trimming of the Event {event_name} of Video {k} Failed',
               flush=True)

     event_annotation.append([narration_id[i], verb[i], noun[i]])

with open('/data/kide004/repos/VideoMAE/dataset/epic/trim_video/sub.csv', 'w', newline='') as f:
     write = csv.writer(f)
     
     write.writerows(event_annotation)


