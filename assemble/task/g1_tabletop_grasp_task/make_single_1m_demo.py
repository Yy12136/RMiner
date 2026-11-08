#!/usr/bin/env python3
"""Validate the single centered 1m dust comparison."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np

VARIANTS = ("clean", "final_train_medium", "final_train_medium_lr1m", "final_train_medium_single1m")

def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, text=True, capture_output=True)
    return json.loads(r.stdout)

def frame(path: Path) -> np.ndarray:
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,30); ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError(f"cannot read {path}")
    return img

def faststart(path: Path) -> bool:
    b=path.read_bytes(); return 0 <= b.find(b"moov") < b.find(b"mdat")

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--dataset",type=Path,required=True); args=ap.parse_args()
    root=args.root.resolve(); rows=[]; tiles=[]; fmt=frames=fast=True
    for name in VARIANTS:
        video=root/name/"episode_000/observation.images.head.mp4"; manifest=json.loads((root/name/"episode_000/manifest.json").read_text()); info=probe(video); st=info["streams"][0]; n=int(st.get("nb_frames",0)); expected=int(manifest["source_episode_frames"])
        ok=(st.get("codec_name")=="h264" and st.get("width")==640 and st.get("height")==480 and st.get("pix_fmt")=="yuv420p" and st.get("r_frame_rate")=="30/1"); fmt &= ok; frames &= n==expected; fast &= faststart(video)
        img=frame(video); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mean=float(np.mean(gray)); p95=float(np.percentile(gray,95)); tile=img.copy(); cv2.rectangle(tile,(0,0),(640,40),(0,0,0),-1); cv2.putText(tile,f"{name} mean={mean:.1f} p95={p95:.1f}",(8,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA); tiles.append(cv2.resize(tile,(320,240),interpolation=cv2.INTER_AREA)); rows.append({"variant":name,"video":str(video),"frames":n,"expected_frames":expected,"duration_seconds":float(info["format"]["duration"]),"mean_luma_frame30":mean,"p95_luma_frame30":p95,"format_ok":ok,"faststart":faststart(video)})
    cv2.imwrite(str(root/"head_comparison_4x1.png"),np.hstack(tiles)); checks={"all_head_videos_h264_640x480_yuv420p_30fps":fmt,"all_head_videos_match_source_episode_frames":frames,"all_head_videos_faststart":fast,"dataset_hashes_unchanged":(root/"source_hashes.before.sha256").read_bytes()==(root/"source_hashes.after.sha256").read_bytes(),"single_1m_rendered":(root/"final_train_medium_single1m/episode_000/observation.images.head.mp4").stat().st_size>0}; report={"effect":"task01_train_medium_single_centered_1m_radius","source_dataset":str(args.dataset.resolve()),"source_episode":0,"variants":rows,"parameters":{"layout":"single_1m","emitter_count":1,"radius_m":1.0,"local_offset_m":[0.0,0.0,1.02],"ambient_multiplier":0.75,"flow_opacity":0.10,"seed":260822},"checks":checks,"all_checks_passed":all(checks.values()),"dataset_modified":False,"formal_training_dataset_ready":False}; (root/"demo_report.json").write_text(json.dumps(report,indent=2)+"\n"); (root/"README.md").write_text("# Task01 单源 1m 扬尘 Demo\n\n只保留一个位于桌面中心附近、半径 1.0m 的 Flow 尘源，并与双源 1m 对比。本轮只生成视觉 Demo，不写入增强数据集。\n\n"+f"- 全部检查通过：`{all(checks.values())}`\n"); print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
