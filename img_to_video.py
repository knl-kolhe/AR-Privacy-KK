# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 06:42:39 2020

@author: Kunal
"""

import os
import ffmpeg


def create_video(Folder, durationMult):
    baselineFolder = "./Videos Dataset/Images/Noise"
    part1Folder = "./Videos Dataset/Images/"+Folder[0]
    part2Folder = "./Videos Dataset/Images/"+Folder[1]
    
    
    folderName = Folder[0]+'-'+Folder[1]
    video_folder = './Videos Dataset/'+folderName+"/"
    mp4_ext = '.mp4'
    if durationMult==1:
        video_name = folderName+'-4pt5min'
    elif durationMult==2:
        video_name = folderName+'-9min' 
    elif durationMult==3:
        video_name = folderName+'-13pt5min'
    else:
        video_name = folderName+'.mp4'
    
    (
        ffmpeg
        .input(baselineFolder+"/*.jpg", pattern_type='glob', framerate=1/30)
        .output(baselineFolder+"/Noise.mp4", s="720x720")
        .overwrite_output()
        .run()
    )
    (
        ffmpeg
        .input(part1Folder+"/*.jpg", pattern_type='glob', framerate=1/30)
        .output(part1Folder+"/"+Folder[0]+mp4_ext, s="720x720")
        .overwrite_output()
        .run()
    )
    (
        ffmpeg
        .input(part2Folder+"/*.jpg", pattern_type='glob', framerate=1/30)
        .output(part2Folder+"/"+Folder[1]+mp4_ext, s="720x720")
        .overwrite_output()
        .run()
    )
    with open(f"./{Folder[0]}-{Folder[1]}-combine.txt",'w') as opfile:
        opfile.write(f"file '{baselineFolder+'/Noise.mp4'}'\nfile '{part1Folder+'/'+Folder[0]+'.mp4'}'\nfile '{part2Folder+'/'+Folder[1]+'.mp4'}'")
        opfile.close()
    
    command = f'ffmpeg -f concat -safe 0 -i {folderName}-combine.txt -c copy "{video_folder+video_name}.mp4" -y'

    os.system(command)
    
#    os.remove(f"./{Folder[0]}-{Folder[1]}-combine.txt")

VideosList = [["Barcode","Text"],["Object","Text"],["Face","Text"]]
for videoName in VideosList:
        create_video(videoName,2)
