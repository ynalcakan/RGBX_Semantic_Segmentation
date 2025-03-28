import os
import glob
import shutil
path = "./Thermal/"

# forget previous task. read train_val.txt and move all files to ssl_RGB/train. If folder not exist, create it
with open('train_val.txt', 'r') as f:
    for line in f:
        line = line.strip() + ".png"
        f_name = os.path.join(path, line)
        print(f_name)
        
        # copy the file to ssl_RGB/train. if folder not exist, create it
        if not os.path.exists('ssl_Thermal/train'):
            os.makedirs('ssl_Thermal/train')
        shutil.copy(f_name, 'ssl_Thermal/train')

