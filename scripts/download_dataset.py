#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 hf-mirror 并行下载 3DGS 数据集的单个场景（images + sparse/）。"""
import os, sys, json, urllib.request, concurrent.futures, time

API = "https://hf-mirror.com/api/datasets/alexmkwizu/gaussian_training_datasets"
RESOLVE = "https://hf-mirror.com/datasets/alexmkwizu/gaussian_training_datasets/resolve/main"
SCENE = sys.argv[1] if len(sys.argv) > 1 else "tandt/truck"
OUT_ROOT = sys.argv[2] if len(sys.argv) > 2 else r"D:\DSH_Project\AI_Projects\data\smoke_truck"
WORKERS = 12

def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def main():
    data = json.loads(fetch(API).decode())
    files = [s["rfilename"] for s in data["siblings"] if s["rfilename"].startswith(SCENE + "/")]
    print(f"[DB] {SCENE}: {len(files)} files", flush=True)

    def dl(f):
        rel = f[len(SCENE) + 1:]
        local = os.path.join(OUT_ROOT, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        url = f"{RESOLVE}/{f}"
        for attempt in range(4):
            try:
                b = fetch(url)
                with open(local, "wb") as fh:
                    fh.write(b)
                return rel, len(b), True
            except Exception as e:
                if attempt == 3:
                    return rel, 0, str(e)
                time.sleep(1 * (attempt + 1))

    done = 0
    total_bytes = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rel, nbytes, ok in ex.map(dl, files):
            done += 1
            if ok is True:
                total_bytes += nbytes
            else:
                print(f"[FAIL] {rel}: {ok}", flush=True)
            if done % 25 == 0 or done == len(files):
                el = time.time() - t0
                print(f"[DB] {done}/{len(files)} | {total_bytes/1e6:.1f} MB | {el:.0f}s | {total_bytes/1e6/el:.2f} MB/s", flush=True)
    print(f"[DB] DONE {SCENE} -> {OUT_ROOT} ({total_bytes/1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
