import os
from aesthetic.aesthetic_inferencer import AestheticInferencer
from aesthetic.maniqa_inferencer import ManiqaInferencer
from aesthetic.musiq_inferencer import MusiqInferencer
from fd.clap_inferencer import ClapInferencer
from dino.dinov3_inferencer import DinoV3Inferencer
from audio_box.audio_box_inferencer import AudioBoxInferencer
from raft.raft_inferencer import RAFTInferencer
from syncformer.syncformer_inferencer import SyncformerInferencer
from syncnet.syncnet_inferencer import SyncnetInferencer
from kl.kld_inferencer import KLDInferencer
from wer.wer_inferencer import WERInferencer
from tqdm import tqdm
from moviepy.editor import *
import json
from PIL import Image
import pickle
import argparse

data_dir = "test_assets"




def evaluate_aesthetic_video(video_path, aesthetic_inferencer, musiq_inferencer, manica_inferencer):
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames())
    aesthetic_scores = []
    musiq_scores = []
    manica_scores = []
    for frame in frames:
        frame = Image.fromarray(frame)
        aesthetic_score = aesthetic_inferencer.infer(frame)
        musiq_score = musiq_inferencer.infer(frame)
        manica_score = manica_inferencer.infer(frame)
        aesthetic_scores.append(aesthetic_score)
        musiq_scores.append(musiq_score)
        manica_scores.append(manica_score)
    avg_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores)
    avg_musiq = sum(musiq_scores) / len(musiq_scores)
    avg_manica = sum(manica_scores) / len(manica_scores)
    return avg_aesthetic / 10, avg_musiq / 100, avg_manica


def evaluate_audiobox_wav(wav_path, audio_box_inferencer):
    score = audio_box_inferencer.infer(wav_path)
    return score['CE'], score['CU'], score['PC'], score['PQ']


def evaluate_dinov3_video(video_path, dinov3_inferencer, image_path):
    anchor_feature = dinov3_inferencer.get_feature(Image.open(image_path))
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames())
    frame_cos = []
    for i, frame in enumerate(frames):
        frame = Image.fromarray(frame)
        feature = dinov3_inferencer.get_feature(frame)
        sim = dinov3_inferencer.infer_feature(feature, anchor_feature)
        frame_cos.append(sim)
    avg_frame_cos = sum(frame_cos) / len(frame_cos)
    return avg_frame_cos

def evaluate_raft_video(video_path, raft_inferencer):
    clip = VideoFileClip(video_path)
    frames = list(clip.iter_frames())
    flow_scores = []
    for i in range(len(frames) - 1):
        frame1 = Image.fromarray(frames[i]).convert("RGB")
        frame2 = Image.fromarray(frames[i + 1]).convert("RGB")
        flow_score = raft_inferencer.infer(frame1, frame2)
        if flow_score is not None:
            flow_scores.append(flow_score)
    avg_flow_score = sum(flow_scores) / len(flow_scores) if len(flow_scores) > 0 else -999
    return avg_flow_score


def calculate_metrics(input_dir,  modes_path, sets_root):
    print("Loading Aesthetic models...")
    aesthetic_inferencer = AestheticInferencer(modes_path)
    print("Loading MusiQ model...")
    musiq_inferencer = MusiqInferencer()
    print("Loading Maniqa model...")
    manica_inferencer = ManiqaInferencer(modes_path)
    print("Loading AudioBox model...")
    audio_box_inferencer = AudioBoxInferencer(modes_path)
    print("Loading Syncformer model...")
    syncformer_inferencer = SyncformerInferencer(modes_path)
    print("Loading Syncnet model...")
    syncnet_inferencer = SyncnetInferencer(modes_path)
    print("Loading RAFT model...")
    raft_inferencer = RAFTInferencer(modes_path)
    print("Loading WER model...")
    wer_inferencer = WERInferencer(modes_path)
    print("Loading CLAP model...")
    clap_inferencer = ClapInferencer(modes_path)
    print("Loading KLD model...")
    kld_inferencer = KLDInferencer()
    print("Loading DINOv3 model...")
    dinov3_inferencer = DinoV3Inferencer(modes_path)
    sez = "set1"
    total_aesthetic = []
    total_musiq = []
    total_maniqa = []
    total_fd = []
    total_kl = []
    total_ce = []
    total_cu = []
    total_pc = []
    total_pq = []
    total_cs = []
    total_wer = []
    total_ava = []
    total_lse_c = []
    total_ms = []
    total_id = []
    for file in os.listdir(f"{sets_root}/{sez}"):
        if file.endswith(".json"):
            base_name = file.split(".")[0]
            item = json.load(open(f"{sets_root}/{sez}/{file}"))
            video_path = f"{input_dir}/{base_name}.mp4"
            if os.path.exists(video_path):
                avg_aesthetic, avg_musiq, avg_manica = evaluate_aesthetic_video(video_path, aesthetic_inferencer,
                                                                                musiq_inferencer, manica_inferencer)
                total_aesthetic.append(avg_aesthetic)
                total_musiq.append(avg_musiq)
                total_maniqa.append(avg_manica)
                if item['audio_prompt']:
                    av_offset = syncformer_inferencer.infer(video_path)
                    total_ava.append(av_offset)
                if item['speech_prompt']['text']:
                    sync_score = syncnet_inferencer.infer(video_path)[1]
                    if sync_score is not None:
                        total_lse_c.append(sync_score)
                flow_score = evaluate_raft_video(video_path, raft_inferencer)
                if flow_score is not None:
                    total_ms.append(flow_score)
                dinov3_score = evaluate_dinov3_video(video_path, dinov3_inferencer,
                                                     f"{sets_root}/{sez}/{base_name}.jpg")
                total_id.append(dinov3_score)
            wav_path = f"{input_dir}/{base_name}.wav"
            if os.path.exists(wav_path):
                ce, cu, pc, pq = evaluate_audiobox_wav(wav_path, audio_box_inferencer)
                total_ce.append(ce)
                total_cu.append(cu)
                total_pc.append(pc)
                total_pq.append(pq)
                if item['audio_prompt']:
                    cs = clap_inferencer.infer(wav_path, item['audio_prompt'][0])
                    total_cs.append(cs)
                if item['speech_prompt']['text']:
                    wer = wer_inferencer.infer_audio_text(wav_path, item['speech_prompt']['text'])
                    total_wer.append(wer)

    sez = "set2"
    for file in os.listdir(f"{sets_root}/{sez}"):
        if file.endswith(".json"):
            base_name = file.split(".")[0]
            item = json.load(open(f"{sets_root}/{sez}/{file}"))
            video_path = f"{input_dir}/{base_name}.mp4"
            if os.path.exists(video_path):
                avg_aesthetic, avg_musiq, avg_manica = evaluate_aesthetic_video(video_path, aesthetic_inferencer,
                                                                                musiq_inferencer, manica_inferencer)
                total_aesthetic.append(avg_aesthetic)
                total_musiq.append(avg_musiq)
                total_maniqa.append(avg_manica)
                if item['audio_prompt']:
                    av_offset = syncformer_inferencer.infer(video_path)
                    total_ava.append(av_offset)
                if item['speech_prompt']['text']:
                    sync_score = syncnet_inferencer.infer(video_path)[1]
                    if sync_score is not None:
                        total_lse_c.append(sync_score)
                flow_score = evaluate_raft_video(video_path, raft_inferencer)
                if flow_score is not None:
                    total_ms.append(flow_score)
                dinov3_score = evaluate_dinov3_video(video_path, dinov3_inferencer,
                                                     f"{sets_root}/{sez}/{base_name}.jpg")
                total_id.append(dinov3_score)

            wav_path = f"{input_dir}/{base_name}.wav"
            if os.path.exists(wav_path):
                ce, cu, pc, pq = evaluate_audiobox_wav(wav_path, audio_box_inferencer)
                total_ce.append(ce)
                total_cu.append(cu)
                total_pc.append(pc)
                total_pq.append(pq)
                if item['audio_prompt']:
                    cs = clap_inferencer.infer(wav_path, item['audio_prompt'][0])
                    fd = clap_inferencer.infer_fd(wav_path, f"{sets_root}/{sez}/{base_name}.wav")
                    kl = kld_inferencer.infer(wav_path, f"{sets_root}/{sez}/{base_name}.wav")
                    total_kl.append(kl)
                    total_cs.append(cs)
                    total_fd.append(fd)
                if item['speech_prompt']['text']:
                    wer = wer_inferencer.infer_audio_text(wav_path, item['speech_prompt']['text'])
                    total_wer.append(wer)

    sez = "set3"
    for file in os.listdir(f"{sets_root}/{sez}"):
        if file.endswith(".json"):
            base_name = file.split(".")[0]
            item = json.load(open(f"{sets_root}/{sez}/{file}"))
            video_path = f"{input_dir}/{base_name}.mp4"
            if os.path.exists(video_path):
                avg_aesthetic, avg_musiq, avg_manica = evaluate_aesthetic_video(video_path, aesthetic_inferencer,
                                                                                musiq_inferencer, manica_inferencer)
                total_aesthetic.append(avg_aesthetic)
                total_musiq.append(avg_musiq)
                total_maniqa.append(avg_manica)
                if item['speech_prompt']['text']:
                    sync_score = syncnet_inferencer.infer(video_path)[1]
                    if sync_score is not None:
                        total_lse_c.append(sync_score)
                flow_score = evaluate_raft_video(video_path, raft_inferencer)
                if flow_score is not None:
                    total_ms.append(flow_score)
                dinov3_score = evaluate_dinov3_video(video_path, dinov3_inferencer,
                                                     f"{sets_root}/{sez}/{base_name}.jpg")
                total_id.append(dinov3_score)
            wav_path = f"{input_dir}/{base_name}.wav"
            if os.path.exists(wav_path):
                if item['speech_prompt']['text']:
                    wer = wer_inferencer.infer_audio_text(wav_path, item['speech_prompt']['text'])
                    total_wer.append(wer)
    fd = sum(total_fd) / len(total_fd) if len(total_fd) > 0 else -999
    kl = sum(total_kl) / len(total_kl) if len(total_kl) > 0 else -999
    id_score = sum(total_id) / len(total_id) if len(total_id) > 0 else -999
    ce = sum(total_ce) / len(total_ce) if len(total_ce) > 0 else -999
    cu = sum(total_cu) / len(total_cu) if len(total_cu) > 0 else -999
    pc = sum(total_pc) / len(total_pc) if len(total_pc) > 0 else -999
    pq = sum(total_pq) / len(total_pq) if len(total_pq) > 0 else -999
    cs = sum(total_cs) / len(total_cs) if len(total_cs) > 0 else -999
    wer = sum(total_wer) / len(total_wer) if len(total_wer) > 0 else -999
    ava = sum(total_ava) / len(total_ava) if len(total_ava) > 0 else -999
    lse_c = sum(total_lse_c) / len(total_lse_c) if len(total_lse_c) > 0 else -999
    ms = sum(total_ms) / len(total_ms) if len(total_ms) > 0 else -999
    avg_aesthetic = sum(total_aesthetic) / len(total_aesthetic) if len(total_aesthetic) > 0 else -999
    avg_musiq = sum(total_musiq) / len(total_musiq) if len(total_musiq) > 0 else -999
    avg_maniqa = sum(total_maniqa) / len(total_maniqa) if len(total_maniqa) > 0 else -999
    avg_as = (avg_aesthetic + avg_musiq + avg_maniqa) / 3.0 if (avg_aesthetic!=-999 and avg_musiq!=-999 and avg_maniqa!=-999) else -999
    return {
        "ms": ms,
        "as": avg_as,
        "id": id_score,
        "fd":fd,
        "kl":kl,
        "cs":cs,
        "ce":ce,
        "cu":cu,
        "pc":pc,
        "pq":pq,
        "wer":wer,
        "lse_c":lse_c,
        "ava":ava,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="mini_testset")
    parser.add_argument("--verse_bench_dir", type=str, default="verse_bench")
    if "MODELS_PATH" in os.environ:
        models_path = os.environ['MODELS_PATH']
    else:
        models_path = "models"
        os.environ['MODELS_PATH'] = models_path

    args = parser.parse_args()
    metrics = calculate_metrics(args.input_dir, models_path, args.verse_bench_dir)
    print(metrics)
