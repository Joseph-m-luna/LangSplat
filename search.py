#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt

#other imports
import torch
import open_clip
from autoencoder.model import Autoencoder
import torch.nn.functional as F

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)
        
        #decoder
        autoencoder_path = "/home/joseph/PRISM/colmap_data/house/ckpt/house_dataset/best_ckpt.pth"
        encoder_hidden_dims = (256, 128, 64, 32, 3)
        decoder_hidden_dims = (16, 32, 64, 128, 256, 256, 512)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

        embedding = clip_lookup("car")
        print("embedding size")
        new_embedding = embedding.repeat(64, 1)

        model.load_state_dict(torch.load(autoencoder_path))

        #batchify rendering
        rendering = rendering.flatten(1)
        chunks = torch.split(rendering, 64, dim=1)

        #process chunks
        proc_chunks = []
        for chunk in chunks:
            flag = False
            if chunk.size(1) < 64:
                new_embedding = embedding.repeat(int(chunk.size(1)), 1)
                print(embedding.size())
            decoded = F.cosine_similarity(model.decode(chunk.transpose(1, 0)), new_embedding, dim=1)
            proc_chunks.append(decoded)
        
        #reconstruct image
        final = torch.cat(proc_chunks).view(737, 981).cpu().numpy()
        print(final.min(), final.max())
        #val = (final.min() + final.max())/2.0 + 0.02
        #val = final.max() - 0.05
        val = 0
        print(val)
        final[final < val] = 0.0
        plt.imshow(final)
        plt.show()

        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        #print(output.keys())
        #print(output["language_feature_image"].size())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

def search():
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print(args)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)

def clip_lookup(text):
    clip_model_type = "ViT-B-16"
    clip_model_pretrained = 'laion2b_s34b_b88k'
    clip_n_dims = 512
    model, _, _ = open_clip.create_model_and_transforms(
        clip_model_type,
        pretrained=clip_model_pretrained,
        precision="fp16",
    )
    model.eval()
    
    tokenizer = open_clip.get_tokenizer(clip_model_type)
    model = model.to("cuda:0")

    tokens = tokenizer(text).to("cuda:0")
    #print(dir(tokens))

    with torch.no_grad():
        embedding = model.encode_text(tokens)

    return embedding

def np_lookup():
    seg = "/home/joseph/PRISM/colmap_data/house/language_features/frame_00006_s.npy"
    feature = "/home/joseph/PRISM/colmap_data/house/language_features/frame_00006_f.npy"
    
    seg_np = np.load(seg)
    feature_np = np.load(feature)
    
    print(feature_np.shape)
    print()
    print()

    print(seg_np[0, :, :][seg_np[0, :, :] != -1.0].min())
    print(seg_np[1, :, :][seg_np[1, :, :] != -1.0].min())
    print(seg_np[2, :, :][seg_np[2, :, :] != -1.0].min())
    print(seg_np[3, :, :][seg_np[3, :, :] != -1.0].min())
    print(seg_np[0, :, :].max())
    print(seg_np[1, :, :].max())
    print(seg_np[2, :, :].max())
    print(seg_np[3, :, :].max())

    print()
    print()

    print(np.unique(seg_np[0, :, :]).size)
    print(np.unique(seg_np[1, :, :]).size)
    print(np.unique(seg_np[2, :, :]).size)
    print(np.unique(seg_np[3, :, :]).size)

    plt.imshow(seg_np[0, :, :])
    plt.show()
    plt.imshow(seg_np[1, :, :])
    plt.show()
    plt.imshow(seg_np[2, :, :])
    plt.show()
    plt.imshow(seg_np[3, :, :])
    plt.show()
    

    #print(np.load(path).shape)
    #print(np.load(path2).shape)
    #print(np.load(path2))
if __name__ == "__main__":
    search()
    #np_lookup()