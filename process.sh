#!/bin/bash

# get the language feature of the scene
python preprocess.py --dataset_name $dataset_path

# train the autoencoder
cd autoencoder
python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ae_ckpt
# e.g. python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

# get the 3-dims language feature of the scene
python test.py --dataset_name $dataset_path --dataset_name $dataset_name
# e.g. python test.py --dataset_path ../data/sofa --dataset_name sofa

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

for level in 1 2 3
do
    python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/$casename/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    # render rgb
    python render.py -m output/${casename}_${level}
    # render language features
    python render.py -m output/${casename}_${level} --include_feature
    # e.g. python render.py -m output/sofa_3 --include_feature
done


#--------------------------------------------------------------------------------
#NOTE: RESIZE IMAGES PRIOR TO RUNNING USING CUSTOM FFMPEG SCRIPT. ALSO PROBABLY USE JPGs
#DATA PREP
#(langsplat) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/LangSplat/preprocess.py --dataset_path . --sam_ckpt_path ~/PRISM/LangSplat/ckpts/sam_vit_h_4b8939.pth --resolution 4
#   - results in 1 new folder
#note - scaled to 1080p, option can be disabled with -r or --resolution 1
#(langsplat) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/LangSplat/autoencoder/train.py --dataset_path . --lr 0.0007 --dataset_name house_dataset
#   - results in 1 new folder, ckpt, which contains another folder (dataset_name), which contains best_ckpt.pth and events.out.tfevents.1711408895.OrangeFalcon.83491.0



#TRAINING 3DGS
#create input folder for converter
#mkdir input
#cp images/* input
#(gaussian_splatting) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/gaussian-splatting/convert.py -s . --resize
#   - results in images_2, images_4, images_8, sparse, stereo, input, run-colmap-geometric.sh, run-colmap-photometric.sh
#(gaussian_splatting) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/gaussian-splatting/train.py -s . --checkpoint_iteration 30000
#   - results in output/e29dfaf9-e


#other stuff
#(langsplat) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/LangSplat/autoencoder/test.py --dataset_name house_dataset --dataset_path .
#   - results in language_features_dim3


#TRAINING ACTUAL MODEL
#(langsplat) joseph@OrangeFalcon:~/PRISM/colmap_data/house$ python ~/PRISM/LangSplat/train.py -s . -m output/model --start_checkpoint output/e29dfaf9-e/chkpnt30000.pth --feature_level 1
#