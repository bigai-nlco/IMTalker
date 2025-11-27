import os
import datetime
import random
import tempfile
import subprocess
import argparse
from typing import Tuple, Optional

import numpy as np
import cv2
import torch
import torchvision
import librosa
import face_alignment
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor

from generator.FM import FMGenerator
from renderer.models import IMTRenderer
from options.base_options import BaseOptions


def load_smirk_params(smirk_data):
    pose = smirk_data["pose_params"].cuda()
    cam = smirk_data["cam"].cuda()
    return pose, cam


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, local_files_only=True
        )

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]

        mult = 360.0 / h

        bboxes = self.fa.face_detector.detect_from_image(img_arr)
        valid_bboxes = [
            (int(x1 ), int(y1), int(x2), int(y2), score)
            for (x1, y1, x2, y2, score) in bboxes if score > 0.95
        ]
        
        if not valid_bboxes:
            raise ValueError("No face detected in the reference image.")

        x1, y1, x2, y2, _ = valid_bboxes[0]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        half_w = int((x2 - x1) * 0.8)
        half_h = int((y2 - y1) * 0.8)
        half = max(half_w, half_h)

        x1_new = max(cx - half, 0)
        x2_new = min(cx + half, w)
        y1_new = max(cy - half, 0)
        y2_new = min(cy + half, h)

        side = min(x2_new - x1_new, y2_new - y1_new)
        x2_new = x1_new + side
        y2_new = y1_new + side

        crop_img = img_arr[y1_new:y2_new, x1_new:x2_new]
        return Image.fromarray(crop_img)

    def default_img_loader(self, path: str) -> Image.Image:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array,
            sampling_rate=sampling_rate,
            return_tensors='pt'
        ).input_values[0]

    def preprocess(self, ref_path: str, audio_path: str, crop: bool) -> dict:
        s = self.default_img_loader(ref_path)
        if crop:
            s = self.process_img(s)
        
        s_tensor = self.transform(s).unsqueeze(0)
        a_tensor = self.default_aud_loader(audio_path).unsqueeze(0)

        return {'s': s_tensor, 'a': a_tensor, 'p': None, 'e': None}


class InferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        
        self.ae = IMTRenderer(self.opt)
        self.fm = FMGenerator(self.opt)
        self._load_models()
        
        self.ae.to(self.rank).eval()
        self.fm.to(self.rank).eval()
        
        self.data_processor = DataProcessor(opt)

    def _load_models(self):
        # Load Renderer
        renderer_ckpt = torch.load(self.opt.renderer_path, map_location="cpu")["state_dict"]
        ae_state_dict = {k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")}
        self.ae.load_state_dict(ae_state_dict, strict=False)

        # Load Generator
        self._load_generator_weights(self.opt.generator_path, self.rank)

    def _load_generator_weights(self, checkpoint_path, rank, prefix='model.'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict:
            state_dict = state_dict['model']

        stripped_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        
        with torch.no_grad():
            for name, param in self.fm.named_parameters():
                if name in stripped_state_dict:
                    param.copy_(stripped_state_dict[name].to(rank))

    def save_video(self, vid_tensor, video_path, audio_path):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_filename = tmp.name
            
            vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
            vid = (vid * 255).type('torch.ByteTensor')
            torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)
            
            if audio_path:
                cmd = f"ffmpeg -i {temp_filename} -i {audio_path} -c:v copy -c:a aac {video_path} -y"
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.rename(temp_filename, video_path)

        if os.path.exists(video_path) and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return video_path

    @torch.no_grad()
    def run_inference(self, res_path, ref_path, aud_path, pose_path=None, gaze_path=None, **kwargs):
        data = self.data_processor.preprocess(ref_path, aud_path, crop=kwargs.get('crop', False))
        
        if pose_path and os.path.exists(pose_path):
            data["pose"], data["cam"] = load_smirk_params(torch.load(pose_path))
        else:
            data["pose"], data["cam"] = None, None

        if gaze_path and os.path.exists(gaze_path):
            data["gaze"] = torch.tensor(np.load(gaze_path), dtype=torch.float32).cuda()
        else:
            data["gaze"] = None
        
        f_r, t_r, g_r = self.encode_image(data['s'].to(self.opt.rank))
        data["ref_x"] = t_r

        sample = self.fm.sample(data, a_cfg_scale=kwargs.get('a_cfg_scale', 1.0), nfe=kwargs.get('nfe', 10), seed=kwargs.get('seed', 25))
        data_out = self.decode_image(f_r, t_r, sample, g_r)
        
        return self.save_video(data_out["d_hat"], res_path, aud_path)

    @torch.no_grad()
    def encode_image(self, x):
        f, g = self.ae.dense_feature_encoder(x)
        t = self.ae.latent_token_encoder(x)
        return f, t, g
    
    @torch.no_grad()
    def decode_image(self, f_r, t_r, t_c, g_r):
        T = t_c.shape[1]
        ta_r = self.ae.adapt(t_r, g_r)
        m_r = self.ae.latent_token_decoder(ta_r)
        
        d_hat = []
        for t in range(T):
            ta_c = self.ae.adapt(t_c[:, t, ...], g_r)
            m_c = self.ae.latent_token_decoder(ta_c)
            d_hat.append(self.ae.decode(m_c, m_r, f_r))
            
        return {'d_hat': torch.stack(d_hat, dim=1).squeeze()}


class InferenceOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path", type=str, default=None)
        parser.add_argument("--pose_path", type=str, default=None)
        parser.add_argument("--gaze_path", type=str, default=None)
        parser.add_argument('--aud_path', type=str, default=None)
        parser.add_argument('--crop', action='store_true')
        parser.add_argument('--res_video_path', type=str, default=None)
        parser.add_argument('--generator_path', type=str, default="./checkpoints/generator.ckpt")
        parser.add_argument('--res_dir', type=str, default="./results")
        parser.add_argument('--input_root', type=str, default=None)
        parser.add_argument('--renderer_path', type=str, default="./checkpoints/renderer.ckpt")
        return parser


def process_item(agent, ref, aud, name, opt):
    pose_root = getattr(opt, "pose_path", None)
    gaze_root = getattr(opt, "gaze_path", None)
    
    pose = os.path.join(pose_root, f"{name}.pt") if pose_root else None
    gaze = os.path.join(gaze_root, f"{name}.npy") if gaze_root else None
    
    out_path = os.path.join(opt.res_dir, f"{name}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Processing: {name}")
    try:
        agent.run_inference(
            out_path, ref, aud, pose, gaze,
            a_cfg_scale=opt.a_cfg_scale, nfe=opt.nfe, crop=opt.crop, seed=opt.seed
        )
    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == '__main__':
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    
    agent = InferenceAgent(opt)
    os.makedirs(opt.res_dir, exist_ok=True)

    ref_path = getattr(opt, "ref_path", None)
    aud_path = getattr(opt, "aud_path", None)

    if ref_path and aud_path:
        if not os.path.exists(ref_path) or not os.path.exists(aud_path):
            raise FileNotFoundError("Reference or audio file not found.")
        name = os.path.splitext(os.path.basename(ref_path))[0]
        process_item(agent, ref_path, aud_path, name, opt)
        
    elif opt.input_root and os.path.exists(opt.input_root):
        for subdir in sorted(os.listdir(opt.input_root)):
            sub_dir_path = os.path.join(opt.input_root, subdir)
            if not os.path.isdir(sub_dir_path): continue

            r_path, a_path = None, None
            for f in os.listdir(sub_dir_path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')): r_path = os.path.join(sub_dir_path, f)
                elif f.lower().endswith(('.wav', '.mp3', '.m4a')): a_path = os.path.join(sub_dir_path, f)

            if r_path and a_path:
                process_item(agent, r_path, a_path, subdir, opt)
    else:
        print("Usage: Provide --ref_path & --aud_path OR --input_root")



