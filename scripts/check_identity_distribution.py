import sys
sys.path.insert(0, '.')
import numpy as np
from src.data_ingestion import load_config, load_lfw

# Used to identify overrepresented identities in LFW before making the
# data-centric improvement of capping images per identity in pair generation

config = load_config()
splits, _ = load_lfw(config)

# combine all splits to check the full dataset distribution
all_labels = np.concatenate([
    splits["train"][1],
    splits["val"][1],
    splits["test"][1],
])

unique, counts = np.unique(all_labels, return_counts=True)
counts_sorted = np.sort(counts)[::-1]

print(f"Total identities: {len(unique)}")
print(f"Total images:     {len(all_labels)}")
print(f"\nTop 10 most represented identities (image count):")
for i in range(10):
    print(f"  Identity {unique[np.argsort(counts)[::-1][i]]}: {counts_sorted[i]} images")

print(f"\nBottom 10 least represented identities:")
for i in range(10):
    print(f"  Identity {unique[np.argsort(counts)[i]]}: {counts_sorted[-(i+1)]} images")

print(f"\nDistribution summary:")
print(f"  Max images per identity: {counts.max()}")
print(f"  Min images per identity: {counts.min()}")
print(f"  Mean images per identity: {counts.mean():.1f}")
print(f"  Identities with only 1 image: {(counts == 1).sum()}")
print(f"  Identities with 2-5 images:   {((counts >= 2) & (counts <= 5)).sum()}")
print(f"  Identities with 6-20 images:  {((counts >= 6) & (counts <= 20)).sum()}")
print(f"  Identities with 20+ images:   {(counts > 20).sum()}")
