zip -r 8_16_first . -x='.git/*' -x='*.pt' -x='*.zip'

import os 
os.chdir('/kaggle/input/8-16-1')
print(os.getcwd())

!cp -r /kaggle/input/8-16-1 /kaggle/working

os.chdir('/kaggle/working/8-16-1')
print(os.getcwd())
!python3 multitask_classifier.py  --use_gpu