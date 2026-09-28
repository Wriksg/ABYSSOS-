import random, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def spot_check():
    tiles = Path("tests/fixtures/empty_tiles.txt").read_text().splitlines()
    sample = random.sample(tiles, min(20, len(tiles)))
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    for ax, t_name in zip(axes.flatten(), sample):
        # FIX: Removed the extra .npz 
        img = np.clip(np.load(f"abyssos_data/train/{t_name}")['hr'][:3].transpose(1,2,0) * 2.5, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        
    plt.suptitle("Eyeball Check: Are these actually empty fields/water?", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": spot_check()
