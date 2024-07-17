import json
import os
import pickle
import time

import cv2
import torch

from encoder import encode_sentences
from InternVid.viclip import frames2tensor, get_viclip, get_vid_feat

model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': 'tool_models/viCLIP/ViClip-InternVid-10M-FLT.pth',
    }
}

class SegmentFeature:
    def __init__(self, video_path_list, base_dir='preprocess', frame_based=False):
        self.video_path_list = sorted(list(set(video_path_list)))
        self.base_dir = base_dir
        self.seconds_per_feat = 2
        self.frames_per_feat = 10

        if frame_based:
            self.create_visual_embedding = self._create_visual_embedding_framebased
        else:
            self.create_visual_embedding = self._create_visual_embedding_videobased
        self.frame_based = frame_based

    def create_textual_embedding(self):
        """use the sentence encoder model to embed the captions of all the videos"""
        model='text-embedding-3-large'
        for video_path in self.video_path_list:
            start_time = time.time()
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            with open(os.path.join(video_dir, 'captions_xcomposer.json')) as f:
                captions = json.load(f)
            caps = list(captions.values())
            caption_emb = encode_sentences(sentence_list=caps, model_name=model)
            print(caption_emb)
            fn = 'segment_textual_embedding_xcomposer.pkl' if self.frame_based else 'segment_textual_embedding_lavila.pkl'
            with open(os.path.join(video_dir, fn), 'wb') as f:
                pickle.dump(caption_emb, f)
            end_time = time.time()
            print(f"textual encoding time for video {base_name}: {round(end_time-start_time, 3)} seconds")

    def _create_visual_embedding_framebased(self):
        start_time = time.time()
        cfg = model_cfgs['viclip-l-internvid-10m-flt']
        model = get_viclip(cfg['size'], cfg['pretrained'])
        assert(type(model)==dict and model['viclip'] is not None and model['tokenizer'] is not None)
        clip, tokenizer = model['viclip'], model['tokenizer']
        clip = clip.to("cuda")
        end_time = time.time()
        print(f'time for loading viCLIP model: {round(end_time-start_time, 3)} seconds')

        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            cap = cv2.VideoCapture(video_path)
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            if 'hm3d' in video_path:
                assert fps == 2
            elif 'scannet' in video_path:
                assert fps == 20
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_feats = total_frames//(fps*self.seconds_per_feat)

            segment_feats = []
            start_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for segment_id in range(total_feats):
                interval = fps * self.seconds_per_feat
                if segment_id == 0:
                    for _ in range(interval // 2): # skip the leading frames
                        cap.read()
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for _ in range(interval): # skip frames in between
                    cap.read()
                frames_tensor = frames2tensor([frame], device='cuda', fnum=1)
                with torch.no_grad():
                    vid_feat = get_vid_feat(frames_tensor, clip).cpu()
                segment_feats.append(vid_feat)
            segment_feats = torch.cat(segment_feats, dim=0).numpy()
            end_time = time.time()
            cap.release()
            print(segment_feats)
            print(f"visual embedding time for video {base_name}: {round(end_time-start_time, 3)} seconds")
            with open(os.path.join(video_dir, 'segment_visual_embedding_frame.pkl'), 'wb') as f:
                pickle.dump(segment_feats, f)

    def _create_visual_embedding_videobased(self):
        start_time = time.time()
        cfg = model_cfgs['viclip-l-internvid-10m-flt']
        model = get_viclip(cfg['size'], cfg['pretrained'])
        assert(type(model)==dict and model['viclip'] is not None and model['tokenizer'] is not None)
        clip, tokenizer = model['viclip'], model['tokenizer']
        clip = clip.to("cuda")
        end_time = time.time()
        print(f'time for loading viCLIP model: {round(end_time-start_time, 3)} seconds')

        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            cap = cv2.VideoCapture(video_path)
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = fps*self.seconds_per_feat//self.frames_per_feat
            total_feats = total_frames//(fps*self.seconds_per_feat)

            segment_feats = []
            start_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for segment_id in range(total_feats):
                frames = []
                for i in range(self.frames_per_feat):
                    success, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    for j in range(frame_interval-1): #skip other frames
                        success, frame = cap.read()
                for i in range(fps*self.seconds_per_feat-frame_interval*self.frames_per_feat):
                    success, frame = cap.read() #skip remaining frames
                frames_tensor = frames2tensor(frames, device='cuda')
                with torch.no_grad():
                    vid_feat = get_vid_feat(frames_tensor, clip).cpu()
                segment_feats.append(vid_feat)
            segment_feats = torch.cat(segment_feats, dim=0).numpy()
            end_time = time.time()
            cap.release()
            print(segment_feats)
            print(f"visual embedding time for video {base_name}: {round(end_time-start_time, 3)} seconds")
            with open(os.path.join(video_dir, 'segment_visual_embedding_video.pkl'), 'wb') as f:
                pickle.dump(segment_feats, f)


    def run(self):
        self.create_textual_embedding()
        self.create_visual_embedding()
