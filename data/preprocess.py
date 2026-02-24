import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

_DIR = Path(__file__).parent

# MovieLens dataset registry
# Each entry: (download_url, subdir_in_zip, ratings_filename, parse_func_name)
# ---------------------------------------------------------------------------
# Parsers: each returns list of (user_id, item_id, timestamp)
# ---------------------------------------------------------------------------


def _parse_ml100k(text: str) -> list[tuple[int, int, int]]:
    """ml-100k: tab-separated (user_id \\t item_id \\t rating \\t timestamp)."""
    interactions = []
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        uid, iid, _, ts = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
        interactions.append((uid, iid, ts))
    return interactions


def _parse_ml1m(text: str) -> list[tuple[int, int, int]]:
    """ml-1m / ml-10m: :: separated (UserID::MovieID::Rating::Timestamp)."""
    interactions = []
    for line in text.strip().split("\n"):
        parts = line.split("::")
        uid, iid, _, ts = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
        interactions.append((uid, iid, ts))
    return interactions


def _parse_ml_csv(text: str) -> list[tuple[int, int, int]]:
    """ml-20m / ml-25m: CSV with header (userId,movieId,rating,timestamp)."""
    interactions = []
    reader = csv.reader(io.StringIO(text))
    next(reader)  # skip header
    for row in reader:
        uid, iid, _, ts = int(row[0]), int(row[1]), row[2], int(float(row[3]))
        interactions.append((uid, iid, ts))
    return interactions


_MOVIELENS_DATASETS = {
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ratings_file": "ml-100k/u.data",
        "parser": _parse_ml100k,
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ratings_file": "ml-1m/ratings.dat",
        "parser": _parse_ml1m,
    },
    "ml-10m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "ratings_file": "ml-10M100K/ratings.dat",
        "parser": _parse_ml1m,  # same :: format as 1m
    },
    "ml-20m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "ratings_file": "ml-20m/ratings.csv",
        "parser": _parse_ml_csv,
    },
    "ml-25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ratings_file": "ml-25m/ratings.csv",
        "parser": _parse_ml_csv,
    },
}

AVAILABLE_DATASETS = list(_MOVIELENS_DATASETS.keys())


# ---------------------------------------------------------------------------
# 5-core filtering
# ---------------------------------------------------------------------------


def _five_core_filter(
    interactions: list[tuple[int, int, int]], min_interactions: int = 5
) -> list[tuple[int, int, int]]:
    """Iteratively remove users/items with fewer than min_interactions until stable."""
    prev_len = -1
    while len(interactions) != prev_len:
        prev_len = len(interactions)

        user_counts: dict[int, int] = defaultdict(int)
        item_counts: dict[int, int] = defaultdict(int)
        for uid, iid, _ in interactions:
            user_counts[uid] += 1
            item_counts[iid] += 1

        interactions = [
            (uid, iid, ts)
            for uid, iid, ts in interactions
            if user_counts[uid] >= min_interactions
            and item_counts[iid] >= min_interactions
        ]

    return interactions


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_and_extract(dataset_name: str) -> str:
    """Download zip, extract ratings file, return its text content."""
    info = _MOVIELENS_DATASETS[dataset_name]
    raw_dir = _DIR / "raw"
    zip_path = raw_dir / f"{dataset_name}.zip"

    # Check if already extracted
    extracted_path = raw_dir / info["ratings_file"]
    if extracted_path.exists():
        return extracted_path.read_text(encoding="utf-8", errors="replace")

    # Download
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading {dataset_name} from {info['url']} ...")
        urlretrieve(info["url"], zip_path)
        print(f"Downloaded to {zip_path}")

    # Extract ratings file only
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(info["ratings_file"], raw_dir)
    print(f"Extracted {info['ratings_file']}")

    return extracted_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def data_load(dataset_name: str):
    """Download (if needed), parse, 5-core filter, remap IDs, save to processed txt.

    Supported datasets: ml-100k, ml-1m, ml-10m, ml-20m, ml-25m

    Output format: one line per interaction, "user_id item_id\\n"
    IDs are remapped starting from 1. Sorted by (user_id, timestamp).
    """
    if dataset_name not in _MOVIELENS_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    output_dir = _DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_data.txt"

    if output_path.exists():
        print(f"{output_path} already exists, skipping.")
        return

    # 1. Download & parse
    info = _MOVIELENS_DATASETS[dataset_name]
    text = _download_and_extract(dataset_name)
    parser = info["parser"]
    interactions = parser(text)
    print(f"Parsed {len(interactions):,} raw interactions")

    # 2. 5-core filtering
    interactions = _five_core_filter(interactions)
    print(f"After 5-core filtering: {len(interactions):,} interactions")

    # 3. Sort by (user, timestamp)
    interactions.sort(key=lambda x: (x[0], x[2]))

    # 4. Remap IDs from 1
    item_set = sorted(set(i[1] for i in interactions))
    item_map = {old: new for new, old in enumerate(item_set, 1)}

    user_set = sorted(set(i[0] for i in interactions))
    user_map = {old: new for new, old in enumerate(user_set, 1)}

    # 5. Save
    with open(output_path, "w") as f:
        for uid, iid, _ts in interactions:
            f.write(f"{user_map[uid]} {item_map[iid]}\n")

    n_users = len(user_set)
    n_items = len(item_set)
    print(
        f"Saved to {output_path}: "
        f"{n_users:,} users, {n_items:,} items, {len(interactions):,} interactions"
    )
