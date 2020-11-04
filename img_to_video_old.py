# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 06:42:39 2020

@author: Kunal
"""

import cv2
import os
#from moviepy.editor import VideoFileClip
import ffmpeg


def create_video(Folder, durationMult):
    video_folder = './Videos Dataset/'+Folder+'/'
    image_folder = video_folder+"Images"
    mp4_ext = '.mp4'
    if durationMult==1:
        video_name = Folder+'-4pt5min'
    elif durationMult==2:
        video_name = Folder+'-9min' 
    elif durationMult==3:
        video_name = Folder+'-13pt5min'
    else:
        video_name = Folder+'.mp4'
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images)
    '''
    print(video_name)
    video = cv2.VideoWriter(video_folder+video_name + avi_ext, cv2.VideoWriter_fourcc(*'mp4v'), 1, (720,720))
    
    def process_img(temp):
        temp = cv2.resize(temp,(720,720))
        temp = cv2.rotate(temp,cv2.ROTATE_90_COUNTERCLOCKWISE)    
        return temp
    
    for image in images:
        seconds = 15*durationMult
        for i in range(seconds):
            temp = cv2.imread(os.path.join(image_folder, image))
            temp = process_img(temp)
            video.write(temp)
    
    cv2.destroyAllWindows()
    video.release()
    '''
    (
        ffmpeg
        .input(image_folder+"/*.jpg", pattern_type='glob', framerate=1/30)
        .output(video_folder+video_name+mp4_ext, s="720x720")
        .overwrite_output()
        .run()
    )
    #for i in range(100):
    #    stream = ffmpeg.input(video_folder+video_name+avi_ext, r='1')
    #    stream = ffmpeg.output(stream, video_folder+video_name+mp4_ext, '-y')
    #    ffmpeg.run(stream)

    # clip = VideoFileClip(video_folder+video_name)
    # clip.write_videofile(video_folder+video_name, codec='libx264')#, audio=True, audio_codec='libfdk_aac')
    # clip.close()

VideosList = ["Barcode-Text","Object-Text","Face-Text"]
for videoName in VideosList:
    for i in range(2,3):
        create_video(videoName,i)
