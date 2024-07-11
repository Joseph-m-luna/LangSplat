import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from io import BytesIO
import matplotlib.pyplot as plt
from autoencoder.model import Autoencoder
import open_clip
from tqdm import tqdm
import torch.nn.functional as F
import time


class LanguageSplat(nn.Module):
    def __init__(self, path):
        super(LanguageSplat, self).__init__()

        if path.endswith(".pth"):
            checkpoint = torch.load(path)
            self.iters = checkpoint[1]

            (self.active_sh_degree, 
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.opt_dict,
                self.spatial_lr_scale) = checkpoint[0]
            
            #embedding = clip_lookup(self.text_input.text())
            autoencoder_path = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/ckpt/neighborhood_encoder_L6/best_ckpt.pth"
            encoder_hidden_dims = (256, 128, 64, 32, 6)
            decoder_hidden_dims = (16, 32, 64, 128, 256, 256, 512)
            self.model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

            #new_embedding = embedding.repeat(64, 1)

            self.model.load_state_dict(torch.load(autoencoder_path))
            self.model.to("cuda:0")

            #get clip models
            clip_model_type = "ViT-B-16"
            clip_model_pretrained = 'laion2b_s34b_b88k'
            clip_n_dims = 512
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_type,
                pretrained=clip_model_pretrained,
                precision="fp16",
            )
            self.clip_model.eval()
            
            self.tokenizer = open_clip.get_tokenizer(clip_model_type)
            self.clip_model = self.clip_model.to("cuda:0")
        elif path.endswith(".ply"):
            autoencoder_path = "/home/joseph/PRISM/colmap_data/neighborhood_corrected/ckpt/neighborhood_encoder_L6/best_ckpt.pth"
            encoder_hidden_dims = (256, 128, 64, 32, 6)
            decoder_hidden_dims = (16, 32, 64, 128, 256, 256, 512)
            self.model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

            self.model.load_state_dict(torch.load(autoencoder_path))
            self.model.to("cuda:0")

            clip_model_type = "ViT-B-16"
            clip_model_pretrained = 'laion2b_s34b_b88k'
            clip_n_dims = 512
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                clip_model_type,
                pretrained=clip_model_pretrained,
                precision="fp32",
            )
            self.clip_model.eval()
            
            self.tokenizer = open_clip.get_tokenizer(clip_model_type)
            self.clip_model = self.clip_model.to("cuda:0")
        
    def save_torch(self, savepath):
        torch.save(((self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._language_feature,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.opt_dict,
                self.spatial_lr_scale), self.iters), savepath)

    def test_colormap(self):
        steps = 1000
        colors = np.zeros((steps//10, steps, 3)).astype(np.uint8)
        start_val = -3.0
        val = start_val
        max_val = 4.0
        i = 0
        while val < max_val:
            if i >= steps:
                break

            colors[:, i] = self.get_color(val=val)
            val += (max_val - start_val)/steps
            i += 1
            
        plt.imshow(colors)
        plt.show()

    def clip_lookup(self, text):
        tokens = self.tokenizer(text).to("cuda:0")
        #print(dir(tokens))

        with torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)

        return embedding
    
    def clip_lookups_cpu(self, text):
        self.clip_model.to("cpu")

        start_time = time.time()
        tokens = self.tokenizer(text).to(torch.int64)
        with torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)
        print("time on cpu:", time.time() - start_time)

        self.clip_model.to("cuda:0")
        return embedding
    
    def get_similarities(self, lang_input="lawns"):
        embedding = self.clip_lookup(lang_input)
        new_embedding = embedding.repeat(64, 1)

        #iterate
        self._similarities = self._features_dc.clone()
        n = 0

        chunks = torch.split(self._language_feature, 64, dim=0)
        proc_chunks = []
        print(self._features_dc.min(), self._features_dc.max())
        for chunk in tqdm(chunks):
            if chunk.size(0) < 64:
                del new_embedding
                new_embedding = embedding.repeat(int(chunk.size(0)), 1)
            decoded = self.model.decode(chunk)
            similarity = F.cosine_similarity(decoded, new_embedding, dim=1).detach().cpu()
            proc_chunks.append(similarity.clone())
            del decoded, chunk, similarity
        i = 0
        #print(proc_chunks)
        combined = torch.cat(proc_chunks)
        val_minimum = combined.min()
        val_maximum = combined.max()
        self.heatmap = []
        for i in tqdm(range(len(combined))):
            rgb = self.get_color(float(combined[i].cpu().numpy()), val_min=val_minimum, val_max=val_maximum, dtype=np.float32)
            self.heatmap.append(rgb)
        self.heatmap = np.array(self.heatmap)
        
        self.export_lang("/home/joseph/Downloads/lawns.splat")
    
    def construct_list_of_attributes_lang(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._language_feature.shape[1]):
            l.append('lang_{}'.format(i))
        return l
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        print(dtype_full)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def get_ply(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        #embedding = self._language_feature.detach().cpu().numpy()
        print(rotation)
        print(rotation.shape)
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        return PlyData([el])
    
    def get_lang_ply(self, filepath):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        embedding = self._language_feature.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_lang()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, embedding), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        with open(filepath, "wb") as f:
            PlyData([el]).write(f)
        return PlyData([el])
    
    def import_lang_ply(self, filepath):
        data = PlyData.read(filepath)
        return data

    def get_color(self, val, val_min=-3.0, val_max=4.0, color_min=0, color_max=255, dtype=np.uint8):
        val = ((val - val_min) / (val_max - val_min))
        red = (val * 2) * (color_max - color_min) + color_min
        if red > color_max:
            red = color_max
        elif red < color_min:
            red = color_min
        red = dtype(red)

        green = (0.5 - abs(val - 0.5)) * 2 * (color_max - color_min) + color_min#dtype((val/2 + val_max/4) * (color_max - color_min) + color_min)
        if green < 0:
            green = 0
        elif green > color_max:
            green = color_max
        green = dtype(green)

        blue = ((1 - 2*val)) * (color_max - color_min) + color_min
        if blue > color_max:
            blue = color_max
        elif blue < color_min:
            blue = color_min
        blue = dtype(blue)
        rgb = np.array([red, green, blue])

        return rgb
    
    def ply_to_langsplat(self, input_file, output_path, lookup_embedding):
        # self.lang_embeddings = []
        # for i in range(len(self._language_feature)):
        #     vec = self._language_feature[i].detach().cpu().numpy()
        #     self.lang_embeddings.append(vec)
        #     break
    
        # self.lang_embeddings = np.array(self.lang_embeddings)
        plydata = self.import_lang_ply(input_file)
        
        lang_0 = plydata['vertex'].data['lang_0']
        lang_1 = plydata['vertex'].data['lang_1']
        lang_2 = plydata['vertex'].data['lang_2']
        lang_3 = plydata['vertex'].data['lang_3']
        lang_4 = plydata['vertex'].data['lang_4']
        lang_5 = plydata['vertex'].data['lang_5']

        chunk_size = 64

        num_chunks = (len(lang_0) + chunk_size - 1) // chunk_size

        car_output = np.empty_like(lang_0, dtype='f4')
        house_output = np.empty_like(lang_0, dtype='f4')
        tree_output = np.empty_like(lang_0, dtype='f4')
        grass_output = np.empty_like(lang_0, dtype='f4')
        road_output = np.empty_like(lang_0, dtype='f4')
        lamp_output = np.empty_like(lang_0, dtype='f4')


        torch.no_grad()

        car_embedding = self.clip_lookup("car, truck, vehicle, bus, automobile, RV, van")
        house_embedding = self.clip_lookup("house roof apartment, building, home, residence")
        tree_embedding = self.clip_lookup("tree, bush, shrub, plant, forest, woods, jungle")
        grass_embedding = self.clip_lookup("grass, lawn, field, meadow, pasture, prairie, yard")
        road_embedding = self.clip_lookup("road, pavement, street, cracked asphalt, trail, sidewalk")
        lamp_embedding = self.clip_lookup("metal pole, street light, lamp post, lighting")

        timestart = time.time()
        result = torch.tensor([], dtype=torch.float32).to("cuda:0")
        print("starting")
        for i in tqdm(range(num_chunks)):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(lang_0))

            #handle end case
            if end - start < 64:
                print("Uneven end case, check for broken code before continuing")
            else:
                lang_0_chunk = lang_0[start:end]
                lang_1_chunk = lang_1[start:end]
                lang_2_chunk = lang_2[start:end]
                lang_3_chunk = lang_3[start:end]
                lang_4_chunk = lang_4[start:end]
                lang_5_chunk = lang_5[start:end]
                
                lang_chunk_numpy = torch.tensor(np.vstack((lang_0_chunk, lang_1_chunk, lang_2_chunk, lang_3_chunk, lang_4_chunk, lang_5_chunk)).T, dtype=torch.float32).to("cuda:0")
                #print(lang_chunk_numpy.shape)

                decoding = self.model.decode(lang_chunk_numpy)
                #car_output[start:end] = F.cosine_similarity(decoding, car_embedding, dim=1).detach().cpu().numpy()
                #house_output[start:end] = F.cosine_similarity(decoding, house_embedding, dim=1).detach().cpu().numpy()
                #tree_output[start:end] = F.cosine_similarity(decoding, tree_embedding, dim=1).detach().cpu().numpy()
                #grass_output[start:end] = F.cosine_similarity(decoding, grass_embedding, dim=1).detach().cpu().numpy()
                #road_output[start:end] = F.cosine_similarity(decoding, road_embedding, dim=1).detach().cpu().numpy()
                #lamp_output[start:end] = F.cosine_similarity(decoding, lamp_embedding, dim=1).detach().cpu().numpy()
                
                result = torch.cat((result, decoding), dim=0)

                #print(lang_output[start:end])
                #print(start, end)
        print("time taken: ", time.time() - timestart)
        print(len(result))
        torch.save(result, "/home/joseph/Downloads/embedding_tensor.pt")
        new_dtype = plydata['vertex'].data.dtype.descr + [('lang_output_car', 'f4')] + [('lang_output_house', 'f4')] + [('lang_output_tree', 'f4')] + [('lang_output_grass', 'f4')] + [('lang_output_road', 'f4')] + [('lang_output_lamp', 'f4')]
        new_vertex_data = np.empty(plydata['vertex'].data.shape, dtype=new_dtype)

        for name in plydata['vertex'].data.dtype.names:
            new_vertex_data[name] = plydata['vertex'].data[name]

        new_vertex_data['lang_output_car'] = car_output
        new_vertex_data['lang_output_house'] = house_output
        new_vertex_data['lang_output_tree'] = tree_output
        new_vertex_data['lang_output_grass'] = grass_output
        new_vertex_data['lang_output_road'] = road_output
        new_vertex_data['lang_output_lamp'] = lamp_output

        vertex_element = PlyElement.describe(new_vertex_data, 'vertex')

        plydata = PlyData([vertex_element])


        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            #color = np.append(self.get_color(v["lang_output_0"], val_min=0, val_max=1, color_min=0, color_max=255, dtype=np.float32), (1 / (1 + np.exp(-v["opacity"])))*255)
            color = (color * 255).clip(0, 255).astype(np.uint8)
            lang = np.array(#TODO: Make dynamic to fit variable language embedding sizes
                [
                    v["lang_output_car"],
                    v["lang_output_house"],
                    v["lang_output_tree"],
                    v["lang_output_grass"],
                    v["lang_output_road"],
                    v["lang_output_lamp"],
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
            buffer.write(lang.astype(np.float32).tobytes())

        self.save_file(buffer.getvalue(), output_path)
    
    def raw_binary_to_comparison(self, bin_path, search_term):
        torch.tensor.from_file(bin_path, dtype=torch.float32)

    def self_to_splat(self, output_path):
        '''
            output_path: Path/filename to save the outout to

            description: Takes as input the current data stored in self and saves it to a .splat file format
        '''
        plydata = self.get_ply()

        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()

        exists = False

        minimum = 0
        maximum = 0
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            rot = ((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8)
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            color = (color * 255).clip(0, 255).astype(np.uint8)
            
            if not exists:
                minimum = color.min()
                maximum = color.max()
                exists = True
            else:
                if color.min() < minimum:
                    minimum = color.min()
                if color.max() > maximum:
                    maximum = color.max()
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write(color.tobytes())
            buffer.write(rot.tobytes())

        print (minimum, maximum)
        self.save_file(buffer.getvalue(), output_path)

    def self_to_lang_ply(self, output_path):
        '''
            output_path: Path to save the current data

            description: Takes as input the current data in self and saves a copy of it to a .ply file format, including language embeddings
        '''
        self.lang_embeddings = []
        for i in range(len(self._language_feature)):
            vec = self._language_feature[i].detach().cpu().numpy()
            self.lang_embeddings.append(vec)
            break
    
        self.lang_embeddings = np.array(self.lang_embeddings)
        plydata = self.get_lang_ply()

        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            color = (color * 255).clip(0, 255).astype(np.uint8)
            lang = np.array(#TODO: Make dynamic to fit variable language embedding sizes
                [
                    v["lang_0"],
                    v["lang_1"],
                    v["lang_2"],
                    v["lang_3"],
                    v["lang_4"],
                    v["lang_5"],
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
            buffer.write(lang.astype(np.float32).tobytes())

        #self.save_file(buffer.getvalue(), output_path)


    def save_file(self, data, output_path):
        with open(output_path, "wb") as f:
            f.write(data)

    def forward(self, camera):
        pass

def pth_to_ply():
    splat = LanguageSplat("/home/joseph/PRISM/colmap_data/neighborhood_corrected/output/neighborhood_corrected_3/chkpnt100000.pth")
    #splat.get_similarities()
    #splat.test_colormap()
    #splat.process_ply_to_splat("/home/joseph/Downloads/model.splat")

    '''export lang ply with lang embedding and sphericals'''
    splat.self_to_lang_ply("/home/joseph/Downloads/neighborhood_layer_1.langsplat")
    #splat.save_ply("/home/joseph/Downloads/model.ply")
    #splat.save("/home/joseph/PRISM/colmap_data/neighborhood_corrected/output/neighborhood_corrected_1/embedded_chkpnt")

def corrected_ply_to_multicolor():
    splat = LanguageSplat("/home/joseph/Downloads/intermediary_corrected.ply")
    splat.ply_to_langsplat("/home/joseph/Downloads/intermediary_corrected.ply", "/home/joseph/Downloads/neighborhood_all_embeddinds.langsplat", "n/a")

def clip_speed_test():
    splat = LanguageSplat("/home/joseph/Downloads/intermediary_corrected.ply")
    start_time = time.time()
    embedding = splat.clip_lookup("car, truck, vehicle, bus, automobile, RV, van")
    print("time on gpu:", time.time() - start_time)

    splat.clip_lookups_cpu("car, truck, vehicle, bus, automobile, RV, van")

def read_speed():
    splat = LanguageSplat("/home/joseph/Downloads/intermediary_corrected.ply")
    start_time = time.time()
    tensor_data = torch.load("/home/joseph/Downloads/embedding_tensor.pt")
    print("time to read tensor data: ", time.time() - start_time)   

    embedding = splat.clip_lookup("car, truck, vehicle, bus, automobile, RV, van")

    #split tensor data into chunks of 512 and run cosine similarity comparison to embedding
    chunk_size = 512
    print(tensor_data.shape, embedding.shape)
    combo_embedding = embedding.repeat(len(tensor_data), 1)

    print(combo_embedding.shape)
    print(tensor_data.shape)
    start_time = time.time()
    similarities = F.cosine_similarity(tensor_data, combo_embedding, dim=1)
    print("time to compute similarities: ", time.time() - start_time)
    print(similarities.shape)
    # for i in tqdm(range(len(tensor_data))):
    #     chunk = tensor_data[i:i+chunk_size]
    #     similarities = torch.cosine_similarity(chunk, embedding.unsqueeze(0), dim=1)
    #     # Perform further processing with the similarities
def server_read_speed():
    import requests
    import io

    filename = "embedding_tensor.pt"
    url = f"https://us-west2-noted-bliss-415620.cloudfunctions.net/getURL?filename={filename}"

    start_time = time.time()
    response = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=10)
    if response.status_code == 200:
        # Process the response if needed
        print(response.json()["data"]["url"])  # or response.text, depending on the response format
    else:
        print(f"Request failed with status code {response.status_code}")
    print("time to get url: ", time.time() - start_time)
    start_time = time.time()
    response = requests.get(response.json()["data"]["url"], timeout=100)
    print(response.json())
    print("time to get data: ", time.time() - start_time)

def clip_cpu_test(input_string="car, truck, vehicle, bus, automobile, RV, van"):
    clip_model_type = "ViT-B-16"
    clip_model_pretrained = 'laion2b_s34b_b88k'
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_type,
        pretrained=clip_model_pretrained,
        precision="fp64",
    )
    print(clip_model)
    
    tokenizer = open_clip.get_tokenizer(clip_model_type)
    clip_model = clip_model.to("cpu")
    tokens = tokenizer(input_string)
    output = clip_model.encode_text(tokens)
    print(output.shape)

    #get the similarity splats
    full_embedding = None

    return output

def upload(bucket_name, source_file_name, destination_blob_name):
    from gcloud import storage

    storage_client = storage.Client()

if __name__ == "__main__":
    #corrected_ply_to_multicolor()
    #pth_to_ply()
    #clip_speed_test()
    #read_speed()
    #server_read_speed()
    #clip_cpu_test()
    upload("prism_intelligence_splats", "/home/joseph/Downloads/embedding_tensor.pt", "embedding_tensor.pt")