# IWTN
# datasets
use *download_cyclegan_dataset.sh* to download *summer2winter* dataset  

the INIT dataset is published on [here](https://zhiqiangshen.com/projects/INIT/index.html)  

the Night2day dataset is on [kaggle](https://www.kaggle.com/datasets/raman77768/day-time-and-night-time-road-images)  

# train
use *scripts/train_x.sh* to train the model  

if you want to train on summer2winter dataset , make sure the path of dataset is right, no need to change any other files.  

  
If the training data set is modified, please modify the corresponding content in *network.py* and *sc_model.py*
specifically modify line 137-155 in *sc_model.py*


