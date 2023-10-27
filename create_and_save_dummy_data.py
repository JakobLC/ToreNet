import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import get_dummy_batch

def main():    
    num_dummy_datapoints = 32
    batch = get_dummy_batch(batch_size=num_dummy_datapoints)

    #save data in ./data/features, ./data/images, ./data/ground_truth

    matplotlib_palette = [0,0,0]+sum([[int(round(c2*255)) for c2 in c] for c in plt.get_cmap("tab20").colors][::2],[])
    folder_names = ["features", "images", "ground_truth"]
    for i in range(num_dummy_datapoints):
        for j in range(len(batch)):
            save_path = os.path.join("./data", folder_names[j], f"{i:06d}.png")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            mult = 255 if j < 2 else 1
            img = Image.fromarray((batch[j][i][0]*mult).astype(np.uint8))
            #put pallete if ground truth
            if j == 2:
                img = img.convert("P", colors=3)
                img.putpalette(matplotlib_palette)
            img.save(save_path)

if __name__=="__main__":
    main()