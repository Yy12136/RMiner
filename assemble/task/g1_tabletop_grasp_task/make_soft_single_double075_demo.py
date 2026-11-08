#!/usr/bin/env python3
"""Validate the softer single-1m and 0.75m dual-source comparison."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np

VARIANTS=("clean","final_train_medium","final_train_medium_single1m_soft","final_train_medium_double075")
def probe(p:Path)->dict:
    r=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate","-show_entries","format=duration","-of","json",str(p)],check=True,text=True,capture_output=True); return json.loads(r.stdout)
def frame(p:Path)->np.ndarray:
    c=cv2.VideoCapture(str(p)); c.set(cv2.CAP_PROP_POS_FRAMES,30); ok,x=c.read(); c.release()
    if not ok: raise RuntimeError(str(p))
    return x
def fast(p:Path)->bool:
    b=p.read_bytes(); return 0<=b.find(b"moov")<b.find(b"mdat")
def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--dataset",type=Path,required=True); a=ap.parse_args(); root=a.root.resolve(); rows=[]; tiles=[]; fmt=frames=fast_ok=True
    for name in VARIANTS:
        p=root/name/"episode_000/observation.images.head.mp4"; man=json.loads((root/name/"episode_000/manifest.json").read_text()); info=probe(p); st=info["streams"][0]; n=int(st.get("nb_frames",0)); exp=int(man["source_episode_frames"]); ok=(st.get("codec_name")=="h264" and st.get("width")==640 and st.get("height")==480 and st.get("pix_fmt")=="yuv420p" and st.get("r_frame_rate")=="30/1"); fmt &= ok; frames &= n==exp; fast_ok &= fast(p); img=frame(p); g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mean=float(np.mean(g)); p95=float(np.percentile(g,95)); t=img.copy(); cv2.rectangle(t,(0,0),(640,40),(0,0,0),-1); cv2.putText(t,f"{name} mean={mean:.1f} p95={p95:.1f}",(8,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA); tiles.append(cv2.resize(t,(320,240),interpolation=cv2.INTER_AREA)); rows.append({"variant":name,"video":str(p),"frames":n,"expected_frames":exp,"duration_seconds":float(info["format"]["duration"]),"mean_luma_frame30":mean,"p95_luma_frame30":p95,"format_ok":ok,"faststart":fast(p)})
    cv2.imwrite(str(root/"head_comparison_4x1.png"),np.hstack(tiles)); checks={"all_head_videos_h264_640x480_yuv420p_30fps":fmt,"all_head_videos_match_source_episode_frames":frames,"all_head_videos_faststart":fast_ok,"dataset_hashes_unchanged":(root/"source_hashes.before.sha256").read_bytes()==(root/"source_hashes.after.sha256").read_bytes(),"soft_single_and_double075_rendered":(root/"final_train_medium_single1m_soft/episode_000/observation.images.head.mp4").stat().st_size>0 and (root/"final_train_medium_double075/episode_000/observation.images.head.mp4").stat().st_size>0}; report={"effect":"task01_soft_single_1m_and_double_075m_dust","source_dataset":str(a.dataset.resolve()),"source_episode":0,"variants":rows,"parameters":{"single_layout":"single_1m_soft","single_emitter_count":1,"single_radius_m":1.0,"single_ambient_multiplier":0.60,"single_flow_opacity":0.055,"double_layout":"left_right_075m","double_emitter_count":2,"double_radius_m":0.75,"double_ambient_multiplier":0.68,"double_flow_opacity":0.12,"seed":260822},"checks":checks,"all_checks_passed":all(checks.values()),"dataset_modified":False,"formal_training_dataset_ready":False}; (root/"demo_report.json").write_text(json.dumps(report,indent=2)+"\n"); (root/"README.md").write_text("# Task01 低浓度单源与 0.75m 双源扬尘 Demo\n\n单源版本保持 1.0m 半径，环境层使用允许范围下限 0.60，并降低 Flow 不透明度；双源版本改为两个半径 0.75m 的尘源。本轮只生成视觉 Demo。\n\n"+f"- 全部检查通过：`{all(checks.values())}`\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
