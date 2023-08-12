# IWTN Instance-aware Weather Translation Network with RoI restricted Transformer bottleneck  
Unpaired image-to-image translation finds extensive applications in domains such
as art, design, and scientific simulations. A notable breakthrough in this field
is CycleGAN, which focus on establishing one-to-one mappings between two
unpaired image domains using generative adversarial networks (GANs) combined
with the cycle-consistency constraint. In more recent works, the emphasis has
shifted towards achieving instance-level translation to enhance the quality of
translated images. However, existing methods often struggle with effectively han-
dling weather translation across various scenes, particularly when dealing with
street view images containing vehicles and traffic signs. Such translations can lead
to unexpected deformations. To address this challenging problem, we propose a
new model that combines CNNs and RoI restricted Transformer architecture. Our
innovative framework leverages the powerful RoI restricted Transformer bottle-
neck, invented the acquisition of larger content features through the self-attention
module, which enables the generator to generate more detailed and realistic
images. Moreover, we introduce an ingenious feature fusion method based on
RoI Align, which changes the way of effective integration of global information
and instance-level information. Additionally, during the feature extraction stage,
our framework employs Squeeze-and-Excitation module, enabling our model to
intelligently adapt and learn intricate channel correlations. We conduct extensive
experiments on publicly available datasets to evaluate the performance of our pro-
posed method. The results demonstrate that our method achieves state-of-the-art
performance in I2I translation within the weather domain.  

# datasets
use *download_cyclegan_dataset.sh* to download *summer2winter* dataset  

the INIT dataset is published on [here](https://zhiqiangshen.com/projects/INIT/index.html)  

the Night2day dataset is on [kaggle](https://www.kaggle.com/datasets/raman77768/day-time-and-night-time-road-images)  

# train
use *scripts/train_x.sh* to train the model  

if you want to train on summer2winter dataset , make sure the path of dataset is right, no need to change any other files.  

  
If the training data set is modified, please modify the corresponding content in *network.py* and *sc_model.py*
specifically modify line 137-155 in *sc_model.py*


