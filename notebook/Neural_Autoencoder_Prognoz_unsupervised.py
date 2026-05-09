import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import json
import re
import shutil
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from pyproj import CRS
from scipy.ndimage import label as nd_label
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# ПАРАМЕТРЫ
# =========================
CELL_SIZE = 500
RANDOM_STATE = 42

# Нейросеть
LATENT_DIM = 6
HIDDEN_DIMS = [48, 24]
DROPOUT = 0.10
NOISE_STD = 0.04
BATCH_SIZE = 1024
MAX_EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
VALID_FRAC = 0.15
USE_GPU_IF_AVAILABLE = True

# Постобработка
SMOOTH_PASSES = 3
N_DISPLAY_CLASSES = 20
TOP_ZONE_Q = 0.993
TOP_ZONE_LOCAL_Q = 0.97
LOCAL_PEAK_SIZE = 5
MIN_TOP_ZONE_CELLS = 2
SHOW_POINTS = False


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_STATE)


def find_base_dir() -> Path:
    candidates = [
        Path.cwd(),
        Path("/mnt/data/prog_zip"),
        Path("/mnt/data"),
        Path(r"C:\Users\janfi\OneDrive\Desktop\Прочее\Прогноз"),
    ]
    for base in candidates:
        shp_dir = base / "shp_dbf"
        if shp_dir.exists() and (shp_dir / "svita_new.shp").exists():
            return base

    for zip_path in [Path("/mnt/data/Прогноз.zip"), Path.cwd() / "Прогноз.zip"]:
        if zip_path.exists():
            unzip_dir = Path("/mnt/data/prog_zip") if str(zip_path).startswith("/mnt/data") else Path.cwd() / "prog_zip"
            unzip_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(unzip_dir)
            shp_dir = unzip_dir / "shp_dbf"
            if shp_dir.exists() and (shp_dir / "svita_new.shp").exists():
                return unzip_dir

    raise FileNotFoundError("Не найдена папка с shp_dbf и svita_new.shp")


BASE_DIR = find_base_dir()
SHP_DIR = BASE_DIR / "shp_dbf"
OUT_DIR = BASE_DIR / "nn_autoencoder_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_ALIAS_DIR = OUT_DIR / "_aliases"
TMP_ALIAS_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("SHP_DIR:", SHP_DIR)
print("OUT_DIR:", OUT_DIR)


def read_sidecar_proj4(shp_path: Path):
    sidecar = shp_path.with_name(shp_path.stem + "_shp.pj4")
    if sidecar.exists():
        txt = sidecar.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"pj4=(.+)", txt)
        if m:
            return m.group(1).strip()
    return None


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    try:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            try:
                from shapely.validation import make_valid
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
            except Exception:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    except Exception:
        pass
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def load_layer(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = repair_geometries(gdf)
    if gdf.crs is None:
        proj4 = read_sidecar_proj4(path)
        if proj4:
            gdf = gdf.set_crs(CRS.from_proj4(proj4), allow_override=True)
    return gdf


def to_crs_safe(gdf: gpd.GeoDataFrame, target_crs):
    if gdf.crs is None and target_crs is not None:
        return gdf.set_crs(target_crs, allow_override=True)
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


def prepare_ascii_aliases(shp_dir: Path, alias_dir: Path):
    aliases, stems = {}, {}
    for name_b in os.listdir(os.fsencode(shp_dir)):
        if not name_b.endswith((b".shp", b".shx", b".dbf", b".prj", b".pj4")) or name_b.endswith(b"_shp.pj4"):
            continue
        base_b, ext_b = os.path.splitext(name_b)
        stems.setdefault(base_b, set()).add(ext_b)

    alias_idx = 0
    for base_b, exts in sorted(stems.items()):
        try:
            base_s = os.fsdecode(base_b)
            safe = all(ord(ch) < 128 and (ch.isalnum() or ch in "_-. ") for ch in base_s)
        except Exception:
            safe = False
            base_s = None

        if safe:
            aliases[base_s] = shp_dir / f"{base_s}.shp"
            continue

        alias = f"layer_{alias_idx:02d}"
        alias_idx += 1
        for ext_b in exts:
            src = os.path.join(os.fsencode(shp_dir), base_b + ext_b)
            dst = alias_dir / f"{alias}{os.fsdecode(ext_b)}"
            shutil.copyfile(src, dst)
        aliases[alias] = alias_dir / f"{alias}.shp"
    return aliases


def normalize_01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    mn = np.nanmin(arr[finite])
    mx = np.nanmax(arr[finite])
    if np.isclose(mx, mn):
        return np.full_like(arr, 0.5, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    out[finite] = (arr[finite] - mn) / (mx - mn)
    return out


def robust_normalize_01(values, q_low=0.02, q_high=0.98):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    lo = np.nanquantile(arr[finite], q_low)
    hi = np.nanquantile(arr[finite], q_high)
    if not np.isfinite(lo):
        lo = np.nanmin(arr[finite])
    if not np.isfinite(hi):
        hi = np.nanmax(arr[finite])
    if hi <= lo:
        return normalize_01(arr)
    clipped = np.clip(arr, lo, hi)
    return normalize_01(clipped)


def build_grid(mask: gpd.GeoDataFrame, cell_size: float):
    mask = repair_geometries(mask)
    mask_union = unary_union(mask.geometry)
    prepared_mask = prep(mask_union)

    minx, miny, maxx, maxy = mask.total_bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    rows = []
    cell_id = 0
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            geom = box(x, y, x + cell_size, y + cell_size)
            if prepared_mask.intersects(geom):
                rows.append((cell_id, r, c, geom))
                cell_id += 1

    grid = gpd.GeoDataFrame(
        rows,
        columns=["cell_id", "row", "col", "geometry"],
        geometry="geometry",
        crs=mask.crs,
    )
    return grid, mask_union, (len(ys), len(xs))


def add_distance_feature(grid: gpd.GeoDataFrame, source: gpd.GeoDataFrame, name: str):
    source = repair_geometries(source)
    source_union = unary_union(source.geometry)
    d = np.empty(len(grid), dtype=float)
    for i, geom in enumerate(grid.geometry.values):
        try:
            d[i] = 0.0 if geom.intersects(source_union) else geom.distance(source_union)
        except Exception:
            d[i] = np.nan
    d = np.nan_to_num(d, nan=np.nanmax(d[np.isfinite(d)]) if np.isfinite(d).any() else 0.0)
    grid[f"dist_{name}"] = d
    return grid


def proximity_from_distance(d, scale):
    d = np.asarray(d, dtype=float)
    return 1.0 / (1.0 + d / scale)


def make_regular_array(grid, value_col, shape):
    arr = np.full(shape, np.nan, dtype=float)
    arr[grid["row"].to_numpy(), grid["col"].to_numpy()] = grid[value_col].to_numpy(dtype=float)
    return arr


def write_back_regular_array(grid, arr, out_col):
    grid[out_col] = arr[grid["row"].to_numpy(), grid["col"].to_numpy()]
    return grid


def neighborhood_mean_std(arr, radius=1):
    sums = np.zeros_like(arr, dtype=float)
    sq_sums = np.zeros_like(arr, dtype=float)
    counts = np.zeros_like(arr, dtype=float)
    valid = np.isfinite(arr)
    base = np.where(valid, arr, 0.0)

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            shifted = np.roll(np.roll(base, dr, axis=0), dc, axis=1)
            shifted_valid = np.roll(np.roll(valid.astype(float), dr, axis=0), dc, axis=1)

            if dr > 0:
                shifted[:dr, :] = 0.0
                shifted_valid[:dr, :] = 0.0
            elif dr < 0:
                shifted[dr:, :] = 0.0
                shifted_valid[dr:, :] = 0.0

            if dc > 0:
                shifted[:, :dc] = 0.0
                shifted_valid[:, :dc] = 0.0
            elif dc < 0:
                shifted[:, dc:] = 0.0
                shifted_valid[:, dc:] = 0.0

            sums += shifted
            sq_sums += shifted ** 2
            counts += shifted_valid

    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    var = np.divide(sq_sums, counts, out=np.full_like(sq_sums, np.nan), where=counts > 0) - mean ** 2
    var = np.where(np.isfinite(var) & (var > 0), var, 0.0)
    std = np.sqrt(var)
    return mean, std


def smooth_on_regular_grid(grid, value_col, shape, passes=2):
    arr = make_regular_array(grid, value_col, shape)
    valid = np.isfinite(arr)
    work = arr.copy()

    for _ in range(passes):
        num = np.zeros_like(work, dtype=float)
        den = np.zeros_like(work, dtype=float)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                shifted = np.roll(np.roll(np.where(np.isfinite(work), work, 0.0), dr, axis=0), dc, axis=1)
                shifted_valid = np.roll(np.roll(np.isfinite(work).astype(float), dr, axis=0), dc, axis=1)

                if dr > 0:
                    shifted[:dr, :] = 0.0
                    shifted_valid[:dr, :] = 0.0
                elif dr < 0:
                    shifted[dr:, :] = 0.0
                    shifted_valid[dr:, :] = 0.0

                if dc > 0:
                    shifted[:, :dc] = 0.0
                    shifted_valid[:, :dc] = 0.0
                elif dc < 0:
                    shifted[:, dc:] = 0.0
                    shifted_valid[:, dc:] = 0.0

                num += shifted
                den += shifted_valid

        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.divide(
                num,
                den,
                out=np.full_like(num, np.nan, dtype=float),
                where=den > 0,
            )
        work = np.where(valid, smoothed, np.nan)

    grid[f"{value_col}_smooth"] = work[grid["row"].to_numpy(), grid["col"].to_numpy()]
    return grid


def local_max_mask(grid, value_col, shape, size=5):
    arr = make_regular_array(grid, value_col, shape)
    half = max(1, int(size) // 2)
    peak = np.zeros_like(arr, dtype=bool)
    valid = np.isfinite(arr)

    for r in range(arr.shape[0]):
        r0 = max(0, r - half)
        r1 = min(arr.shape[0], r + half + 1)
        for c in range(arr.shape[1]):
            if not valid[r, c]:
                continue
            c0 = max(0, c - half)
            c1 = min(arr.shape[1], c + half + 1)
            window = arr[r0:r1, c0:c1]
            if np.isfinite(window).any() and arr[r, c] >= np.nanmax(window):
                peak[r, c] = True

    return peak[grid["row"].to_numpy(), grid["col"].to_numpy()]


def connected_component_filter(grid, value_col, shape, min_cells=2):
    arr = np.zeros(shape, dtype=np.uint8)
    rr = grid["row"].to_numpy()
    cc = grid["col"].to_numpy()
    arr[rr, cc] = grid[value_col].to_numpy(dtype=np.uint8)
    structure = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
    labels, n = nd_label(arr, structure=structure)
    keep = np.zeros(shape, dtype=np.uint8)
    for i in range(1, n + 1):
        comp = labels == i
        if comp.sum() >= min_cells:
            keep[comp] = 1
    return keep[rr, cc]


def detect_layers(alias_map):
    names = sorted(alias_map.keys())
    print("Слои:", names)
    required = {
        "mask": ["svita_new"],
        "facies": ["fasii"],
        "paleo": ["gr_dol_vp_poly"],
        "struct": ["kory"],
        "magm": ["dayki_buf"],
        "tect1": ["glub_raz_nw"],
        "tect2": ["glub_r_nw"],
    }
    out = {}
    for key, candidates in required.items():
        hit = None
        for cand in candidates:
            if cand in alias_map:
                hit = cand
                break
        if hit is None:
            raise FileNotFoundError(f"Не найден слой для {key}: ожидался один из {candidates}")
        out[key] = hit
    return out


def detect_point_layers(alias_map, mask_crs):
    candidates = []
    for name, path in alias_map.items():
        try:
            gdf = load_layer(path)
            if gdf.empty:
                continue
            gdf = to_crs_safe(gdf, mask_crs)
            geom_types = sorted(set(gdf.geom_type.astype(str)))
            if all(gt in ["Point", "MultiPoint"] for gt in geom_types):
                candidates.append({"layer": name, "n": len(gdf), "geom_types": geom_types})
        except Exception:
            continue
    return candidates


def collect_evidence_points(alias_map, point_candidates, mask_union, mask_crs):
    name_hits = ["result", "gold", "zolot", "uran", "u", "au"]
    keep = []
    for item in point_candidates:
        lname = item["layer"].lower()
        if item["layer"] == "result" or any(k in lname for k in name_hits):
            keep.append(item["layer"])

    if not keep and point_candidates:
        keep = [x["layer"] for x in sorted(point_candidates, key=lambda z: z["n"], reverse=True)[:2]]

    layers = []
    for name in keep:
        gdf = load_layer(alias_map[name])
        gdf = to_crs_safe(gdf, mask_crs)
        gdf = gdf[gdf.geometry.within(mask_union)].copy()
        if not gdf.empty:
            gdf["source_layer"] = name
            layers.append(gdf[["source_layer", "geometry"]])

    if layers:
        return gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), geometry="geometry", crs=mask_crs)
    return gpd.GeoDataFrame(columns=["source_layer", "geometry"], geometry="geometry", crs=mask_crs)


aliases = prepare_ascii_aliases(SHP_DIR, TMP_ALIAS_DIR)
layer_names = detect_layers(aliases)
print("Используемые слои:", layer_names)

layers = {k: load_layer(aliases[v]) for k, v in layer_names.items()}
mask = layers["mask"]
if mask.crs is None:
    raise ValueError("У маски не определена CRS")

for key in layers:
    layers[key] = to_crs_safe(layers[key], mask.crs)

point_candidates = detect_point_layers(aliases, mask.crs)
print("Точечные слои-кандидаты:", point_candidates)

# =========================
# СЕТКА И ПРИЗНАКИ
# =========================
grid, mask_union, grid_shape = build_grid(layers["mask"], CELL_SIZE)
print("Ячеек в сетке:", len(grid))
print("Размер сетки (rows, cols):", grid_shape)

for name in ["facies", "paleo", "struct", "magm", "tect1", "tect2"]:
    grid = add_distance_feature(grid, layers[name], name)

scales = {
    "facies": 1000,
    "paleo": 1200,
    "struct": 900,
    "magm": 800,
    "tect1": 800,
    "tect2": 800,
}
for name, scale in scales.items():
    grid[f"prox_{name}"] = proximity_from_distance(grid[f"dist_{name}"], scale)

# Базовые взаимодействия без повторения методички как алгоритма;
# это просто инженерные признаки для нейросети.
grid["tect_intersection"] = grid["prox_tect1"] * grid["prox_tect2"]
grid["tect_struct"] = ((grid["prox_tect1"] + grid["prox_tect2"]) / 2.0) * grid["prox_struct"]
grid["tect_magm"] = ((grid["prox_tect1"] + grid["prox_tect2"]) / 2.0) * grid["prox_magm"]
grid["paleo_struct"] = grid["prox_paleo"] * grid["prox_struct"]

base_features = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm", "prox_tect1", "prox_tect2",
    "tect_intersection", "tect_struct", "tect_magm", "paleo_struct",
]

for col in base_features:
    arr = make_regular_array(grid, col, grid_shape)
    neigh_mean, neigh_std = neighborhood_mean_std(arr, radius=1)
    grid = write_back_regular_array(grid, neigh_mean, f"{col}_nbr_mean")
    grid = write_back_regular_array(grid, neigh_std, f"{col}_nbr_std")

feature_cols = base_features + [f"{c}_nbr_mean" for c in base_features] + [f"{c}_nbr_std" for c in base_features]
X = grid[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    med = float(np.nanmedian(X[col])) if np.isfinite(X[col]).any() else 0.0
    X[col] = X[col].fillna(med)

# Небольшая стабилизирующая трансформация для перекошенных распределений
X_trans = X.copy()
for col in X_trans.columns:
    vals = X_trans[col].to_numpy(dtype=float)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    X_trans[col] = np.sqrt(vals)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_trans)
print("Размер матрицы признаков:", X_scaled.shape)

# =========================
# НЕЙРОСЕТЬ: DENOISING AUTOENCODER
# =========================
device = torch.device("cuda" if USE_GPU_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
print("Устройство:", device)

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=6, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def make_dataloaders(X, valid_frac=0.15, batch_size=512, seed=42):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = max(1, int(n * valid_frac))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(X_valid, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, train_idx, valid_idx


def train_autoencoder(X, input_dim):
    model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_loader, valid_loader, train_idx, valid_idx = make_dataloaders(X, valid_frac=VALID_FRAC, batch_size=BATCH_SIZE, seed=RANDOM_STATE)

    best_state = None
    best_valid = np.inf
    patience_left = EARLY_STOPPING_PATIENCE
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb in train_loader:
            xb = xb.to(device)
            noisy = xb + NOISE_STD * torch.randn_like(xb)
            optimizer.zero_grad()
            recon, _ = model(noisy)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb in valid_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                loss = criterion(recon, xb)
                valid_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        valid_loss = float(np.mean(valid_losses)) if valid_losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_valid - 1e-6:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | valid={valid_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_valid


model, history_df, best_valid_loss = train_autoencoder(X_scaled, input_dim=X_scaled.shape[1])
print("Лучший valid loss:", round(float(best_valid_loss), 6))

# Получаем реконструкцию и латентное пространство
with torch.no_grad():
    xt = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    recon_t, latent_t = model(xt)
    recon = recon_t.cpu().numpy()
    latent = latent_t.cpu().numpy()

reconstruction_error = np.mean((X_scaled - recon) ** 2, axis=1)
latent_center = np.median(latent, axis=0)
latent_radius = np.sqrt(((latent - latent_center) ** 2).sum(axis=1))

# Итоговый нейросетевой score: только из выходов нейросети
# reconstruction_error ловит необычные сочетания признаков,
# latent_radius добавляет отделение устойчивых редких режимов в bottleneck-пространстве.
grid["ae_recon_error"] = robust_normalize_01(reconstruction_error, 0.02, 0.995)
grid["ae_latent_radius"] = robust_normalize_01(latent_radius, 0.02, 0.995)
grid["nn_raw"] = 0.78 * grid["ae_recon_error"] + 0.22 * grid["ae_latent_radius"]

for j in range(latent.shape[1]):
    grid[f"latent_{j+1}"] = latent[:, j]

# Пространственное сглаживание итогового score
for col in ["nn_raw", "ae_recon_error", "ae_latent_radius"]:
    grid = smooth_on_regular_grid(grid, col, grid_shape, passes=SMOOTH_PASSES)

grid["prospectivity"] = robust_normalize_01(
    0.82 * grid["nn_raw_smooth"] +
    0.10 * grid["ae_recon_error_smooth"] +
    0.08 * grid["ae_latent_radius_smooth"],
    0.02,
    0.995,
)
grid["prognoz"] = 1.0 - grid["prospectivity"]

# Локальная сила для компактных top-зон
grid["local_strength"] = grid["prospectivity"] - grid["prospectivity"].quantile(0.50)
grid["local_strength"] = robust_normalize_01(grid["local_strength"], 0.05, 0.995)

def mark_top_zones(grid, shape):
    q_main = float(grid["prospectivity"].quantile(TOP_ZONE_Q))
    q_local = float(grid["local_strength"].quantile(TOP_ZONE_LOCAL_Q))
    peak_mask = local_max_mask(grid, "prospectivity", shape, size=LOCAL_PEAK_SIZE)
    grid["top_zone"] = (
        (grid["prospectivity"] >= q_main) &
        (grid["local_strength"] >= q_local) &
        peak_mask
    ).astype(int)
    grid["top_zone"] = connected_component_filter(grid, "top_zone", shape, min_cells=MIN_TOP_ZONE_CELLS).astype(int)
    return grid


grid = mark_top_zones(grid, grid_shape)

# =========================
# ОЦЕНКА ПО ИЗВЕСТНЫМ ТОЧКАМ
# =========================
evidence_points = collect_evidence_points(aliases, point_candidates, mask_union, mask.crs)
print("Количество точек для проверки:", len(evidence_points))

metrics = {}
if not evidence_points.empty:
    joined = gpd.sjoin(
        evidence_points,
        grid[["cell_id", "prospectivity", "prognoz", "top_zone", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.dropna(subset=["cell_id"]).copy()
    if not joined.empty:
        q10 = float(grid["prospectivity"].quantile(0.90))
        q15 = float(grid["prospectivity"].quantile(0.85))
        q20 = float(grid["prospectivity"].quantile(0.80))

        metrics = {
            "n_grid_cells": int(len(grid)),
            "n_evidence_points": int(len(joined)),
            "best_valid_loss": float(best_valid_loss),
            "mean_reconstruction_error": float(np.mean(reconstruction_error)),
            "top10_hit_rate": float((joined["prospectivity"] >= q10).mean()),
            "top15_hit_rate": float((joined["prospectivity"] >= q15).mean()),
            "top20_hit_rate": float((joined["prospectivity"] >= q20).mean()),
            "top_zone_hit_rate": float((joined["top_zone"] == 1).mean()),
            "n_top_zone_cells": int(grid["top_zone"].sum()),
            "top_zone_fraction": float(grid["top_zone"].mean()),
        }
    else:
        metrics = {"warning": "Точки не попали внутрь ячеек сетки"}
else:
    metrics = {"warning": "Нет точек для проверки; модель обучена без учителя только по сетке факторов"}

print(json.dumps(metrics, ensure_ascii=False, indent=2))

# =========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================
grid_centroids = grid.copy()
grid_centroids["geometry"] = grid_centroids.geometry.centroid
csv_cols = [
    "cell_id", "row", "col", "prospectivity", "prognoz", "top_zone",
    "ae_recon_error", "ae_latent_radius", "nn_raw", "local_strength",
] + feature_cols
csv_cols = [c for c in csv_cols if c in grid_centroids.columns]
grid_centroids[csv_cols].to_csv(OUT_DIR / "grid_nn_attributes.csv", index=False, encoding="utf-8-sig")

history_df.to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

params_dump = {
    "CELL_SIZE": CELL_SIZE,
    "LATENT_DIM": LATENT_DIM,
    "HIDDEN_DIMS": HIDDEN_DIMS,
    "DROPOUT": DROPOUT,
    "NOISE_STD": NOISE_STD,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_EPOCHS": MAX_EPOCHS,
    "LEARNING_RATE": LEARNING_RATE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "VALID_FRAC": VALID_FRAC,
    "SMOOTH_PASSES": SMOOTH_PASSES,
    "TOP_ZONE_Q": TOP_ZONE_Q,
    "TOP_ZONE_LOCAL_Q": TOP_ZONE_LOCAL_Q,
    "LOCAL_PEAK_SIZE": LOCAL_PEAK_SIZE,
    "MIN_TOP_ZONE_CELLS": MIN_TOP_ZONE_CELLS,
}
with open(OUT_DIR / "run_params.json", "w", encoding="utf-8") as f:
    json.dump(params_dump, f, ensure_ascii=False, indent=2)

# GPKG
out_gpkg = OUT_DIR / "nn_autoencoder_prospectivity.gpkg"
if out_gpkg.exists():
    out_gpkg.unlink()

grid.to_file(out_gpkg, layer="grid_result", driver="GPKG")
grid[grid["top_zone"] == 1].to_file(out_gpkg, layer="top_zones", driver="GPKG")
if not evidence_points.empty:
    evidence_points.to_file(out_gpkg, layer="evidence_points", driver="GPKG")

# =========================
# ГРАФИКИ
# =========================
history_png = OUT_DIR / "training_history.png"
plt.figure(figsize=(7, 4))
plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
plt.plot(history_df["epoch"], history_df["valid_loss"], label="valid")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Denoising autoencoder training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(history_png, dpi=200, bbox_inches="tight")
plt.close()

png_path = OUT_DIR / "nn_prospectivity_map.png"
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
plot_data = grid.copy()
plot_data.plot(column="prognoz", cmap="RdYlBu_r", ax=ax, linewidth=0, legend=False)

# Белые изолинии по prospectivity
try:
    arr = make_regular_array(grid, "prospectivity", grid_shape)
    masked = np.ma.masked_invalid(arr)
    xs = np.arange(grid_shape[1])
    ys = np.arange(grid_shape[0])
    ax.contour(xs, ys, masked, levels=12, colors="white", linewidths=0.7, alpha=0.7)
except Exception:
    pass

if not evidence_points.empty and SHOW_POINTS:
    evidence_points.plot(ax=ax, color="black", markersize=6, alpha=0.7)

top_zones = grid[grid["top_zone"] == 1]
if not top_zones.empty:
    top_zones.plot(ax=ax, color="#F2D700", edgecolor="black", linewidth=0.3)

legend_handles = [Patch(facecolor="#F2D700", edgecolor="black", label="Top zone")]
ax.legend(handles=legend_handles, loc="lower right")
ax.set_title("Neural network prospectivity (Autoencoder, unsupervised)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(png_path, dpi=250, bbox_inches="tight")
plt.close(fig)

print("Готово. Основной результат:", png_path)
print("GPKG:", out_gpkg)
