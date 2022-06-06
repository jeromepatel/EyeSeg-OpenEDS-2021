# EyeSeg-OpenEDS-2021
3D Eye Segmentation Track 1 OpenEDS 2021, Point Transformer + CNN based method Implementation

## Update:
The reproducible code for the competition (Pointnet2 MSG) is availabe [here](https://github.com/jeromepatel/Eye-Segmentation-3rd-Rank-Winning-Solution). Make sure to check the updated and more reader friendly code, this repo includes almost everything (which is kinda messy). 

#### Note: Eye Segmentation Data was provided by Facebook Research team on request basis, as part of the competition. The competition is over (Ended on Aug 1st, 2021), the dataset is no longer available for download publically.
The competition website is:<br/>
[OpenEDS 2021 facebook-research](https://research.fb.com/programs/facebook-openeds-2021-challenge/) <br/>
[EvalAI,submission & leaderboard website](https://eval.ai/web/challenges/challenge-page/896/overview)
</br>

This repository contains all of my experiments as part of my participation in the OpenEDS 2021 challenge track 1 (this was my task during the 2 month long internship at American University of Sharjah). Our final position is 3rd on [leaderboard](https://eval.ai/web/challenges/challenge-page/896/leaderboard/2362), (NainaNet lol). Apologies for the mess as the Point transformer code is barely runnable on windows 10 (partly due to heavy RAM and GPU requirements, beware of config file, especially `npoints` in Point Cloud, `k` in KNN and `batch_size` during training).  <br/>

The second part is PointNet based architecture which smoothly runs on >= 6 GB GPU RAM with great results. Most of these architecture use CNN or modified versions of Convolution Operations with some nice grouping and sampling tricks. We got good results with PointNet2 (or ++) MSG and DGCNN models. These are easy to train too. 
Currently, this repo exclusively stores all of the code, however I am planning to release neat and clean code which can be run with minimal configuration and setup. Still if you want to run (for your own experiments, please place modify first `dataset.py` file for point transformer, place data in `data/` directory and modify `config` appropriatly, don't forget to change the suitable model architecture for Pointcloud.  <br/>

Code Environment: 
* Python - 3.8.10
* plyfile - 0.7.4
* pytorch - 1.9.0
* CUDA Version - 11.4
* open3d - 0.13.0
* (Also the dataset which is not public anymore)

Majority of my code related to PT is modified from a really [great repo on Point-transformer by Yang You](https://github.com/qq456cvb/Point-Transformers)
Other PointNet based code is borrowed from [here, Pointnet2 Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (coz it's simply awesome and very intuitive for my task). I would like to thank them for opensourcing their implementations as it provided a great headstart for my approach (baseline as well as experiments). 
<br/>
I'll try to clean-up the code and post a new repo with less ugly code, stay tuned and have a great day!

If you have any questions, send an email at makadiyajyot2121@gmail.com
