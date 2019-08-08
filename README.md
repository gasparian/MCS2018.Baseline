
# [MCS2018: Adversarial Attacks on Black-box Face Recognition](https://github.com/AlexanderParkin/MCS2018.Baseline) X Workshop  

*Trained Resnet18 student model included for experiments with white-box attacks*  
*See the main reference for more info*  

## How to reproduce  

Get and run prepared docker image:  
```
docker pull gasparjan/mcs2018:latest
```  
```
docker run -it -p 8888:8888 -v /home/gas/Documents/MCS2018.Baseline:/home/MCS2018.Baseline --ipc=host gasparjan/mcs2018:latest jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/home/MCS2018.Baseline'  
```  

Run `explore.ipynb`.  

## How to improve  
Try other attacker types. [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/pdf/1801.00553.pdf)  
