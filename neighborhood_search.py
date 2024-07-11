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
import cv2
from skimage.morphology import erosion, dilation, closing, opening, area_closing, area_opening
from PIL import Image



#GUI imports
import time
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
import math

import platform

#python ~/PRISM/LangSplat/neighborhood_search.py --model_path output/neighborhood_corrected_3 --include_feature

def getWorld2View2_v2(R, t, camera_rot, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    R_local = np.dot(R, camera_rot)
    Rt[:3, :3] = R_local.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class RenderViewer(QMainWindow):
    def __init__(self, render_func, width, height, R, T, trans, scale, gaussians, pipeline, background, args):
        super().__init__()

        #camera initialization
        self.znear = 0.01
        self.zfar = 100
        self.FoVx = 1.222862
        self.FoVy=0.9656996

        self.world_view = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj = (self.world_view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        view = MiniCam(width=width, height=height, fovy=self.FoVy, fovx=self.FoVx, znear=self.znear, zfar=self.zfar, world_view_transform=self.world_view, full_proj_transform=self.full_proj)

        self.rotation = R
        self.translation = T
        self.trans = trans
        self.scale = scale
        self.view = view
        
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.args = args

        #window_params
        self.render_func = render_func
        self.width = width / 2
        self.height = height / 2
        self.rot = np.eye(3)
        
        self.initUI()
        self.startRenderLoop()
        self.w = False
        self.a = False
        self.s = False
        self.d = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.theta = math.pi/4
        self.ctrl = False
        self.shift = False
        self.q = False
        self.e = False
        self.r = False
        self.f = False

        
    def initUI(self):
        self.setWindowTitle("Render Viewer")

        # Main container widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Main layout
        main_layout = QHBoxLayout(main_widget)

        # Render area
        self.label = QLabel(self)
        self.label.setFixedSize(self.width, self.height)

        # Sidebar
        sidebar = QVBoxLayout()

        # Example sidebar content
        self.text_input = QLineEdit(self)
        self.button = QPushButton("Apply Query", self)
        self.button.clicked.connect(self.onButtonClick)

        # Add a spacer to push items to the bottom
        sidebar.addStretch(1)
        sidebar.addWidget(self.text_input)
        sidebar.addWidget(self.button)
        sidebar.setAlignment(Qt.AlignBottom)

        # Add render area and sidebar to main layout
        main_layout.addWidget(self.label)
        
        # Create a widget for the sidebar with a fixed width
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(200)  # Adjust width as needed

        main_layout.addWidget(sidebar_widget)

    def startRenderLoop(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.render)
        self.timer.start(0)  # Trigger as fast as possible
    
    def render(self):
        speed = 0.035
        rot_speed = 0.1
        
        if self.shift:
            self.translation[1] -= speed
        if self.ctrl:
            self.translation[1] += speed
        if self.w:
            self.trans[0] -= speed
        if self.s:
            self.trans[0] += speed
        if self.a:
            self.translation[0] += speed
        if self.d:
            self.translation[0] -= speed
        if self.q:
            R1 = np.array([[math.cos(rot_speed), -math.sin(rot_speed), 0],
                  [math.sin(rot_speed), math.cos(rot_speed),  0],
                  [0,                   0,                    1]])
            self.rot = np.dot(self.rot, R1)
        if self.e:
            R2 = np.array([[math.cos(-rot_speed), -math.sin(-rot_speed), 0],
                  [math.sin(-rot_speed), math.cos(-rot_speed),  0],
                  [0,                   0,                    1]])
            self.rot = np.dot(self.rot, R2)

        #update camera information
        self.world_view = torch.tensor(getWorld2View2(self.rotation, self.translation, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj = (self.world_view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.view.world_view_transform = self.world_view
        self.view.full_proj_transform = self.full_proj
        
        frame = self.render_func(self.view, self.gaussians, self.pipeline, self.background, self.args)
        data_np = np.array((np.transpose(frame["render"].detach().cpu().numpy(), (1,2,0)))).copy()
        #data_np = data_np * 255.0
        data_np[data_np < 0.0] = 0.0
        data_np[data_np > 1.0] = 1.0

        data_np = (data_np * 255).astype(np.uint8)

        qimage = QImage(data_np.data, data_np.shape[1], data_np.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.width, self.height)  # Resize the pixmap to fit the window
        self.label.setPixmap(pixmap)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.w = True
        elif event.key() == Qt.Key_A:
            self.a = True
        elif event.key() == Qt.Key_S:
            self.s = True
        elif event.key() == Qt.Key_D:
            self.d = True
        elif event.key() == Qt.Key_Up:
            self.up = True
        elif event.key() == Qt.Key_Down:
            self.down = True
        elif event.key() == Qt.Key_Left:
            self.left = True
        elif event.key() == Qt.Key_Right:
            self.right = True
        elif event.key() == Qt.Key_Shift:
            self.shift = True
        elif event.key() == Qt.Key_Control:
            self.ctrl = True
        elif event.key() == Qt.Key_Q:
            self.q = True
        elif event.key() == Qt.Key_E:
            self.e = True
        elif event.key() == Qt.Key_R:
            self.r = True
        elif event.key() == Qt.Key_F:
            self.f = True
        
        # Call the parent class implementation
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_W:
            self.w = False
        elif event.key() == Qt.Key_A:
            self.a = False
        elif event.key() == Qt.Key_S:
            self.s = False
        elif event.key() == Qt.Key_D:
            self.d = False
        elif event.key() == Qt.Key_Up:
            self.up = False
        elif event.key() == Qt.Key_Down:
            self.down = False
        elif event.key() == Qt.Key_Left:
            self.left = False
        elif event.key() == Qt.Key_Right:
            self.right = False
        elif event.key() == Qt.Key_Shift:
            self.shift = False
        elif event.key() == Qt.Key_Control:
            self.ctrl = False
        elif event.key() == Qt.Key_Q:
            self.q = False
        elif event.key() == Qt.Key_E:
            self.e = False
        elif event.key() == Qt.Key_R:
            self.r = False
        elif event.key() == Qt.Key_F:
            self.f = False

    def onButtonClick(self):
        resolution = 300
        range_val = 0.12
        embedding = clip_lookup(self.text_input.text())
        frame = self.render_func(self.view, self.gaussians, self.pipeline, self.background, self.args)
        autoencoder_path = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/ckpt/neighborhood_encoder_L6/best_ckpt.pth"
        encoder_hidden_dims = (256, 128, 64, 32, 6)
        decoder_hidden_dims = (16, 32, 64, 128, 256, 256, 512)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

        new_embedding = embedding.repeat(64, 1)

        model.load_state_dict(torch.load(autoencoder_path))

        #batchify rendering
        start_time = time.time()
        print(frame["language_feature_image"].size())
        rendering = frame["language_feature_image"].flatten(1)
        chunks = torch.split(rendering, 64, dim=1)

        #process chunks
        proc_chunks = []
        for chunk in tqdm(chunks):
            if chunk.size(1) < 64:
                del new_embedding
                new_embedding = embedding.repeat(int(chunk.size(1)), 1)
            decoded = model.decode(chunk.transpose(1, 0))
            #start = torch.cuda.memory_allocated("cuda:0")
            similarity = F.cosine_similarity(decoded, new_embedding, dim=1).detach().cpu()
            #print("similarities:", similarity, F.L1_loss())
            #end = torch.cuda.memory_allocated("cuda:0")
            #Fprint(start, end, end - start)
            proc_chunks.append(similarity.clone())
            del decoded, chunk, similarity
        
        #reconstruct image
        final = torch.cat(proc_chunks).view(int(self.height * 2), int(self.width * 2)).cpu().numpy()
        plt.imshow(final)
        plt.savefig("search.png", dpi=resolution)
        print(final.min(), final.max())
        #val = (final.min() + final.max())/2.0 + 0.02
        #val = final.max() - range_val
        val = 0

        kernel = np.ones((5, 5), np.uint8)
        element = np.array([[0,1,1,1,0],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [0,1,1,1,0]])
        seg = np.array(final > val).astype(np.uint8)
        seg = erosion(seg, element)
        seg = erosion(seg, element)
        seg = erosion(seg, element)
        seg = dilation(seg, element)
        seg = dilation(seg, element)
        seg = dilation(seg, element)
        #seg = cv2.erode(seg, kernel, iterations=1)

        #Overlay segmentation on RGB render
        #rgb_render = view.original_image[0:3, :, :].detach().cpu().numpy()
        rgb_render = frame["render"].detach().cpu().numpy()

        red = rgb_render[0, :, :]
        green = rgb_render[1, :, :]
        blue = rgb_render[2, :, :]
        red = segment(red, seg, final, 1)
        green = segment(green, seg, final, 255)
        blue = segment(blue, seg, final, 255)

        #final[final < val] = 0.0
        rgb_render = np.stack([red, green, blue])
        print("red", red.min(), red.max())
        print("blue", blue.min(), blue.max())
        print("green", green.min(), green.max())

        total_time = time.time()
        print("total time:", total_time)

        plt.imshow(np.transpose(rgb_render, (1, 2, 0)))
        plt.savefig("searchOverlay.png", dpi=resolution)

    def mousePressEvent(self, event):
        print(f"mouse clicked {event}")
        # Clear focus from the text input when clicking outside of it
        self.setFocus()
        super().mousePressEvent(event)

def render_sample():
    time.sleep(0.02)
    data = np.random.uniform(0, 255, size=(981, 737, 3)).astype(np.uint8)
    return data

def start_viewer():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    #print(args)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    #passed:
    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    args = args

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt100000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        print(f"gaussians: {gaussians._language_feature.size()}")
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #if not skip_train:
        #     render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        #if not skip_test:
        #     render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

    #set up renderer
    if False:
        views = scene.getTrainCameras()[0:1]
        for view_set in views:
            for item in dir(view_set):
                if not item.startswith("_"):
                    attr = getattr(view_set, item)
                    print(f"{item} : {type(attr)}")
                    print(f"\t{attr}")
    #time.sleep(1000)
    '''
    world_view = torch.Tensor([[-1.4308e-03,  7.7034e-01, -6.3763e-01,  0.0000e+00],
        [-2.6845e-01,  6.1393e-01,  7.4231e-01,  0.0000e+00],
        [ 9.6329e-01,  1.7223e-01,  2.0592e-01,  0.0000e+00],
        [-2.4111e+00,  9.6320e-01, -9.6460e-01,  1.0000e+00]]).to("cuda:0")
    full_proj = torch.Tensor([[-2.0409e-03,  1.4695e+00, -6.3769e-01, -6.3763e-01],
        [-3.8293e-01,  1.1711e+00,  7.4239e-01,  7.4231e-01],
        [ 1.3741e+00,  3.2854e-01,  2.0594e-01,  2.0592e-01],
        [-3.4392e+00,  1.8373e+00, -9.7469e-01, -9.6460e-01]]).to("cuda:0")
    '''
    
    w=1980
    h=1480
    trans=np.array([0.0, 0.0, 0.0])
    scale=1.0
    R = np.array([[-0.00143077,  0.77034319, -0.63762789],
         [-0.26845035,  0.61392751,  0.74231221],
         [ 0.96329246,  0.17223351,  0.20592051]])
    T = np.array([-2.41107078,  0.96319961, -0.96459507])


    #start viewer
    app = QApplication(sys.argv)
    window = RenderViewer(render, w, h, R, T, trans, scale, gaussians, pipeline, background, args)
    window.show()
    app.exec()


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
        #SEARCH PARAMETERS
        range_val = 0.08
        embedding = clip_lookup("cracked asphalt")


        print(f"rendering view: {view}")
        print(dir(view))
        output = render(view, gaussians, pipeline, background, args)

        rgb_render = output["render"]
        rendering = output["language_feature_image"]
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
        else:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)
        #rendering = gt
        #rendering = gt
        if False:
            plt.imshow(gt.detach().cpu().numpy()[0, :, :])
            plt.show()
            plt.imshow(gt.detach().cpu().numpy()[1, :, :])
            plt.show()
            plt.imshow(gt.detach().cpu().numpy()[2, :, :])
            plt.show()
            plt.imshow(gt.detach().cpu().numpy()[3, :, :])
            plt.show()
            plt.imshow(gt.detach().cpu().numpy()[4, :, :])
            plt.show()
            plt.imshow(gt.detach().cpu().numpy()[5, :, :])
            plt.show()
        if False:
            print("SHOWING")
            print(rendering.size())
            decoded_rendering = rendering.detach().cpu().numpy()
            print(decoded_rendering.shape)
            plt.imshow(np.transpose(gt.detach().cpu().numpy(), (1, 2, 0)))
            plt.show()
        
        #decoder
        autoencoder_path = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/ckpt/neighborhood_encoder_L6/best_ckpt.pth"
        encoder_hidden_dims = (256, 128, 64, 32, 6)
        decoder_hidden_dims = (16, 32, 64, 128, 256, 256, 512)
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

        
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
            decoded = model.decode(chunk.transpose(1, 0))
            similarity = F.cosine_similarity(decoded, new_embedding, dim=1)
            proc_chunks.append(similarity)
        
        #reconstruct image
        final = torch.cat(proc_chunks).view(737, 981).cpu().numpy()
        plt.imshow(final)
        plt.show()
        print(final.min(), final.max())
        #val = (final.min() + final.max())/2.0 + 0.02
        val = final.max() - range_val
        #val = 0

        kernel = np.ones((5, 5), np.uint8)
        element = np.array([[0,1,1,1,0],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [0,1,1,1,0]])
        seg = np.array(final > val).astype(np.uint8)
        seg = erosion(seg, element)
        seg = erosion(seg, element)
        seg = erosion(seg, element)
        seg = dilation(seg, element)
        seg = dilation(seg, element)
        seg = dilation(seg, element)
        #seg = cv2.erode(seg, kernel, iterations=1)

        #Overlay segmentation on RGB render
        #rgb_render = view.original_image[0:3, :, :].detach().cpu().numpy()
        rgb_render = rgb_render.detach().cpu().numpy()

        red = rgb_render[0, :, :]
        green = rgb_render[1, :, :]
        blue = rgb_render[2, :, :]
        red = segment(red, seg, final, 1.5)
        green = segment(green, seg, final, 255)
        blue = segment(blue, seg, final, 1)

        #final[final < val] = 0.0
        rgb_render = np.stack([red, green, blue])
        print("red", red.min(), red.max())
        print("blue", blue.min(), blue.max())
        print("green", green.min(), green.max())

        plt.imshow(np.transpose(rgb_render, (1, 2, 0)))
        plt.show()

        #np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        #np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        #print(output.keys())
        #print(output["language_feature_image"].size())
        #torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def segment(channel, booleans, heatmap, divisor):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap*(1.0/heatmap.max())
    output = []
    for row in range(len(channel)):
        new_row = []
        for column in range(len(channel[row])):
            if booleans[row, column]:
                new_row.append(heatmap[row, column]/divisor)
            else:
                new_row.append(channel[row, column])
        output.append(new_row)
    
    return np.array(output)

               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt100000.pth')
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
    #print(args)
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
    seg = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/language_features/00000_s.npy"
    image = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/images/00000.jpg"
    feature = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/language_features/00000_f.npy"
    
    img = Image.open(image)
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    seg_np = np.load(seg)
    feature_np = np.load(feature)

    print(seg_np.shape)
    print(img.shape)
    plt.imshow(img)
    plt.show()

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

def np_resize():
    dir = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/language_features/"
    files = os.listdir(dir)
    for filename in files:
        if filename.endswith("s.npy"):
            file = np.load(dir + filename)
            layers = []
            for layer in file:
                new_layer = cv2.resize(layer, dsize=(981, 737))
                layers.append(np.array(new_layer))
            file = np.stack(layers)
            np.save(dir + filename, file)
if __name__ == "__main__":
    start_viewer()
    #search()
    #np_resize()
    #np_lookup()