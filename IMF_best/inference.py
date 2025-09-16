import argparse
import os
import torch
from torch.utils import data

from tqdm import tqdm
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch
import torch.nn.functional as F
from torch import nn, optim
from train_swin import IMFSystem
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import cv2
import numpy as np
from dataset import VFHQ, VFHQ_test, gghead_test, Settingtest
# 在IMFSystem类中添加加载方法
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 1)
    vid = vid.clamp(0, 1).cpu().numpy()
    vid = (vid * 255).astype(np.uint8)
    T, H, W, C = vid.shape

    #vid = np.concatenate((vid), axis=-2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    # 写入每一帧
    for frame in vid:
        # OpenCV 要求输入为 BGR，因此需要从 RGB 转换
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved successfully to {save_path}")
    #print(save_path)
    #torchvision.io.write_video(save_path, vid, fps=fps)

class Demo(nn.Module):
    def __init__(self, args, gen):
        super(Demo, self).__init__()

        self.args = args
        print('==> loading model')
        self.gen = gen.to("cuda")
        self.gen.eval()

        print('==> loading data')
        self.save_path = args.save_folder
        os.makedirs(self.save_path, exist_ok=True) 
        self.dataset_test = VFHQ_test(test_dir=self.args.input_path, device="cuda")
        self.loader_test = data.DataLoader(
            self.dataset_test,
            num_workers=0,
            batch_size=1,
            sampler=None,
            pin_memory=False,
            drop_last=False,
        )
        #self.save_path = os.path.join(self.save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        #self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        #self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        #self.vid_target = self.vid_target.cuda()

    def run(self):

        print('==> running')
        with torch.no_grad():
            pbar = tqdm(range(len(self.dataset_test)), desc="Inferencing Progress")  # 使用 tqdm 创建进度条
            #source = _load_image_256("E:\\codes\\codes\\IMF_new\\examples\\reference_images\\7.jpg").to("cuda").unsqueeze(dim=0)
            loader = sample_data(self.loader_test)
            for idx in pbar:
                batch = next(loader)
                #if batch["video_id"][0] != "video_22":continue
                source = batch["video_frames_256"][0]
                driving = batch["video_frames_256"][0:]
                # 新增可视化视频参数
                #att_frames = []  # 存储可视化帧
                #layer_frames = []
                #output_att_path = os.path.join(self.save_path, f"{video_id[0]}_att.mp4")
                #output_layer_path = os.path.join(self.save_path, f"{video_id[0]}_att.mp4")
                vid_target_recon = []
                for i in tqdm(range(len(driving))):
                    img_target_recon = self.gen(driving[i], source)
                        
                    vid_target_recon.append(img_target_recon)
                vid_target_recon = torch.cat(vid_target_recon, dim=0)
                save_video(vid_target_recon, os.path.join(self.save_path, batch['video_id'][0]+'.mp4'), 25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加GAN相关参数
    parser.add_argument("--use_gan", action='store_true', help="Enable GAN training")
    parser.add_argument("--gan_type", type=str, default='lsgan', choices=['lsgan', 'vanilla'])
    parser.add_argument("--gan_weight", type=float, default=0.1, help="Weight for GAN loss")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--exp_path", type=str, default='./exps')
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_l1", type=float, default=1.0)
    parser.add_argument("--loss_vgg", type=float, default=1.0)
    parser.add_argument("--decoder", type=str, default="frame")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--save_folder", type=str, default='E:\\codes\\codes\\LIA\\result_test\\')
    parser.add_argument("--model_path", type=str, default="E:\codes\codes\IMF_last\exps\\10l1_0.5vgg_1gan_2dpos_10id\checkpoints\last.ckpt")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--r1_reg_every", type=int, default=16, 
                        help="Frequency to apply R1 regularization (e.g., apply every 16 steps)")
    parser.add_argument("--use_r1_reg", action='store_true', help="gan R1 reg")
    parser.add_argument("--use_2dpos", action='store_true', help="2d position encoding")
    parser.add_argument("--use_arcface", action='store_true', help="use arcface loss")
    parser.add_argument("--arcface_path", type=str, default="E:\codes\codes\model_ir_se50.pth", help="2d position encoding")
    parser.add_argument("--loss_arcface", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument('--swin_res_threshold', type=int, default=128, help='Resolution threshold to switch to Swin Attention.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Attention.')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Stochastic depth rate for Swin.')
    parser.add_argument('--low_res_depth', type=int, default=2, help='Number of TransformerBlocks for low-res features.')
    parser.add_argument("--input_path", type=str, default='E:\\codes\\codes\\LIA\\result_test\\')
    
    # 删除下面这行
    # parser = pl.Trainer.add_argparse_args(parser)
    
    # 直接解析参数
    args = parser.parse_args()

    # 训练器参数直接在Trainer初始化时设置
    # 初始化系统
    system = IMFSystem(args)
    # 配置logger和checkpoint
    logger = TensorBoardLogger(save_dir=args.exp_path, name=args.exp_name)
    # 修改主函数部分
    if args.model_path:
        system.load_ckpt(
            args.model_path, 
        )
        print(f"Resumed from checkpoint: {args.model_path}")
    else:
        system = IMFSystem(args)
    
    demo = Demo(args, system.gen)
    demo.run()