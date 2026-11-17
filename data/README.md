# Data

The images are **not** committed to this repository. They come from the official
**Grocery Store Dataset** and are downloaded with a *sparse* git checkout, so only the
12 selected product folders (plus the split files) are fetched.

- Source: <https://github.com/marcusklasson/GroceryStoreDataset>
- License: MIT (see `GroceryStoreDataset/LICENSE` after download)
- Citation: M. Klasson, C. Zhang, H. Kjellström, *A Hierarchical Grocery Store Image Dataset
  with Visual and Semantic Labels*, IEEE WACV 2019.

## Download (PowerShell, from the project root)

```powershell
# 1. Split files only (no images yet)
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/marcusklasson/GroceryStoreDataset.git data\GroceryStoreDataset
git -C data\GroceryStoreDataset sparse-checkout set --no-cone "/dataset/*.txt" "/dataset/classes.csv" "/LICENSE" "/README.md"
git -C data\GroceryStoreDataset checkout

# 2. Exact counts + list of folders to download (writes data/sparse_paths.txt)
python -m src.prepare_data --counts-only

# 3. Images of the 12 selected classes only, then verification (writes data/manifest.csv)
Get-Content data\sparse_paths.txt | git -C data\GroceryStoreDataset sparse-checkout set --no-cone --stdin
python -m src.prepare_data
```

## Files in this folder

| File | Created by | Content |
|---|---|---|
| `GroceryStoreDataset/` | git (ignored by our repo) | Sparse clone of the official dataset |
| `sparse_paths.txt` | `prepare_data --counts-only` | Folders fetched by the sparse checkout |
| `manifest.csv` | `prepare_data` | One row per selected image: split, path, class, label index, size |

The split counts and integrity checks are in `../results/data_audit/`.

The official train / val / test splits are used as-is. Images are never moved between splits.

Dataset size: the original release has 5,125 images (2,640 train + 2,485 test, 81 classes); a
validation split (296 images in `val.txt`) was added by the authors in 2019. This project uses the
780 images of 12 classes: 366 train, 59 val, 355 test.
