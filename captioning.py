import sys

sys.path.insert(0, 'LaViLa/')
import json
import os
import pickle
import time
import urllib.request
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from encoder import load_xcomposer
from LaViLa.eval_narrator import decode_one
from LaViLa.lavila.data.video_transforms import Permute
from LaViLa.lavila.models.models import \
    VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from LaViLa.lavila.models.tokenizer import MyGPT2Tokenizer


class Captioning:
    def __init__(self, video_path_list, base_dir='preprocess', frame_based=False):
        self.video_path_list = sorted(list(set(video_path_list)))
        self.seconds_per_caption = 2 # a caption covers 2 seconds
        self.frames_per_caption = 4 # a caption is generated from 4 frames in the 2-second segments
        self.base_dir = base_dir

        if frame_based:
            self.generate_captions_for_all_videos = self._generate_captions_for_all_videos_framebased
        else:
            self.generate_captions_for_all_videos = self._generate_captions_for_all_videos_videobased

    def _generate_captions_for_all_videos_framebased(self):
        """create the captions for all videos"""
        start_time = time.time()

        model, tokenizer = load_xcomposer()

        end_time = time.time()
        print(f'time for loading captioning model: {round(end_time-start_time, 3)} seconds')

        print(f'Total of {len(self.video_path_list)} videos.')
        for video_path in self.video_path_list:
            print(video_path)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Unable to open video file.")
                continue
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            if 'hm3d' in video_path:
                assert fps == 2
            elif 'scannet' in video_path:
                assert fps == 20

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_captions = total_frames//(fps*self.seconds_per_caption)

            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            captions = dict()
            start_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for caption_id in range(total_captions):
                interval = fps * self.seconds_per_caption
                if caption_id == 0:
                    for _ in range(interval // 2): # skip the leading frames
                        cap.read()
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for _ in range(interval): # skip frames in between
                    cap.read()

                prompt = '<ImageHere>Please describe this image in briefly.'
                cv2.imwrite('/tmp/tmp.png', frame)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        text, _ = model.chat(tokenizer, query=prompt, image='/tmp/tmp.png', history=[], do_sample=True) # do_sample=True seems to do better on hm3d

                caption_start_frame = caption_id*fps*self.seconds_per_caption
                caption_end_frame = (caption_id+1)*fps*self.seconds_per_caption

                segment = "{}_{}".format(str(caption_start_frame), str(caption_end_frame))
                captions[segment] = text

                print(f"id: {caption_id}, frame_interval: {segment}, caption: {text}")
            end_time = time.time()
            cap.release()
            print(f"captioning time for video {base_name}: {round(end_time-start_time, 3)} seconds")
            with open(os.path.join(video_dir, "captions_xcomposer.json"), 'w') as f:
                json.dump(captions, f)
            segments = list(captions)
            segment2id = dict()
            for segment in segments:
                segment2id[segment] = len(segment2id)
            with open(os.path.join(video_dir, "segment2id.json"), 'w') as f:
                json.dump(segment2id, f)

    def _generate_captions_for_all_videos_videobased(self):
        """create the captions for all videos"""
        start_time = time.time()
        crop_size = 336
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
        ])
        ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
        ckpt_path = os.path.join('tool_models/LaViLa/', ckpt_name)
        if not os.path.exists(ckpt_path):
            print('downloading model to {}'.format(ckpt_path))
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name), ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        # instantiate the model, and load the pre-trained weights
        model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
            text_use_cls_token=False,
            project_embed_dim=256,
            gated_xattn=True,
            timesformer_gated_xattn=False,
            freeze_lm_vclm=False,      # we use model.eval() anyway
            freeze_visual_vclm=False,  # we use model.eval() anyway
            num_frames=4,
            drop_path_rate=0.
        )
        model.load_state_dict(state_dict, strict=True)
        model.cuda()
        model.eval()
        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
        end_time = time.time()
        print(f'time for loading captioning model: {round(end_time-start_time, 3)} seconds')


        for video_path in self.video_path_list:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Unable to open video file.")
                continue
            if 'hm3d' in video_path:
                fps = 2
            elif 'scannet' in video_path:
                fps = 20
            else:
                fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_captions = total_frames//(fps*self.seconds_per_caption)
            frame_interval = fps*self.seconds_per_caption//self.frames_per_caption # the interval between two selected frames

            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            captions = dict()
            start_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for caption_id in range(total_captions):
                frames = []
                for i in range(self.frames_per_caption): # 4 frames are selected for generating the caption
                    success, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    for j in range(frame_interval-1): #skip other frames
                        success, frame = cap.read()
                for i in range(fps*self.seconds_per_caption-frame_interval*self.frames_per_caption):
                    success, frame = cap.read() #skip remaining frames
                frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
                frames = torch.stack(frames, dim=0)
                frames = val_transform(frames)
                frames = frames.unsqueeze(0)

                with torch.no_grad():
                    input_frames = frames.cuda(non_blocking=True)
                    image_features = model.encode_image(input_frames)
                    generated_text_ids, ppls = model.generate(
                        image_features,
                        tokenizer,
                        target=None,  # free-form generation
                        max_text_length=77,
                        top_k=None,
                        top_p=0.95,   # nucleus sampling
                        num_return_sequences=5,  # number of candidates: 5
                        temperature=0.7,
                        early_stopping=True,
                    )
                text = ""
                length = -1
                for i in range(5):
                    # select the longest candidate as the caption
                    generated_text_str = decode_one(generated_text_ids[i], tokenizer)
                    if len(generated_text_str) > length:
                        length = len(generated_text_str)
                        text = generated_text_str
                caption_start_frame = caption_id*fps*self.seconds_per_caption
                caption_end_frame = (caption_id+1)*fps*self.seconds_per_caption
                segment = "{}_{}".format(str(caption_start_frame), str(caption_end_frame))
                captions[segment] = text
                print(f"id: {caption_id}, frame_interval: {segment}, caption: {text}")
            end_time = time.time()
            cap.release()
            print(f"captioning time for video {base_name}: {round(end_time-start_time, 3)} seconds")
            with open(os.path.join(video_dir, "captions_lavila.json"), 'w') as f:
                json.dump(captions, f)
            segments = list(captions)
            segment2id = dict()
            for segment in segments:
                segment2id[segment] = len(segment2id)
            with open(os.path.join(video_dir, "segment2id.json"), 'w') as f:
                json.dump(segment2id, f)

    def run(self):
        self.generate_captions_for_all_videos()