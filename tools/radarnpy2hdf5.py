import os
import numpy as np
import torch
import torchvision.transforms as transforms

class Normalize(object):
    def _call_(self, radarData):
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues
        maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataNorm = radarDataZero / (maxValues + 1e-8)
        std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
        return (radarDataNorm - mean.view(c, 1, 1)) / (std.view(c, 1, 1) + 1e-8)

root_in = r"F:\mmpoint_project\preprocess_radar"
root_out = r"D:\Semester 5\vision\vision project\hdf5_ready"

normalize = Normalize()

for folder in sorted(os.listdir(root_in)):
    single_path = os.path.join(root_in, folder, "vert")
    if not os.path.isdir(single_path):
        continue

    target_path = os.path.join(root_out, folder)
    os.makedirs(target_path, exist_ok=True)
    npy_files = sorted([f for f in os.listdir(single_path) if f.endswith(".npy")])

    if not npy_files:
        print(f"No npy files found in {single_path}")
        continue

    print(f"Processing {folder} — {len(npy_files)} frames...")

    for npy_file in npy_files:
        npy_file_path = os.path.join(single_path, npy_file)
        radar_data = np.load(npy_file_path)

        VRDAEmaps = torch.zeros((8, 2, 64, 64, 8))
        idxSampleChirps = 0
        numChirps, numFrames = 16, 8

        for idxChirps in range(numChirps // 2 - numFrames // 2,
                                numChirps // 2 + numFrames // 2):
            real_part = torch.tensor(radar_data[idxChirps].real, dtype=torch.float32)
            imag_part = torch.tensor(radar_data[idxChirps].imag, dtype=torch.float32)
            VRDAEmaps[idxSampleChirps, 0] = normalize(real_part)
            VRDAEmaps[idxSampleChirps, 1] = normalize(imag_part)
            idxSampleChirps += 1

        save_path = os.path.join(target_path, npy_file)
        np.save(save_path, VRDAEmaps.detach().cpu().numpy())

    print(f"Finished {folder} — saved to {target_path}")

print("All radar sequences converted successfully")