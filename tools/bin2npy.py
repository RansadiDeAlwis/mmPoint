import os
import numpy as np
from process_iwr1843 import RadarObject

def bin2npy_stream(bin_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    radar = RadarObject()

    adc = radar.getadcDataFromDCA1000(bin_file)
    print("Shape of radar data:", adc.shape)

    for idx in range(radar.numFrame):
        frame = adc[:, radar.numChirp*idx : radar.numChirp*(idx+1), 0:radar.numADCSamples]
        heatmap = radar.generateHeatmap(frame)

        out_path = os.path.join(save_dir, f"{idx:09d}.npy")
        np.save(out_path, heatmap)

        # progress every 10 frames
        if idx % 10 == 0:
            print(f"[saved] {os.path.basename(save_dir)} frame {idx}")

if __name__ == "__main__":

    radar_root = "radar"                 # your radar/ folder
    save_root  = "preprocess_radar"       # output root
    os.makedirs(save_root, exist_ok=True)

    # scenes to process
    target_singles = [1,2,3, 4, 5, 6, 7, 8, 9, 10]

    # numerical sort ensures 3 → 10 in order
    single_folders = sorted(
        [f for f in os.listdir(radar_root) if f.startswith("single_")],
        key=lambda x: int(x.split("_")[-1])
    )

    for folder in single_folders:
        sid = int(folder.split("_")[-1])
        if sid not in target_singles:
            continue

        print(f"\nProcessing single_{sid} (vertical only)...")

        bin_file = os.path.join(radar_root, f"single_{sid}", "vert", "adc_data.bin")
        save_dir = os.path.join(save_root, f"single_{sid}", "vert")
        os.makedirs(save_dir, exist_ok=True)

        # skip if already converted
        if any(name.endswith(".npy") for name in os.listdir(save_dir)):
            print(f"Skip single_{sid}: {save_dir} already has npy files.")
            continue

        bin2npy_stream(bin_file, save_dir)

        print(f"✅ Done single_{sid}\n")