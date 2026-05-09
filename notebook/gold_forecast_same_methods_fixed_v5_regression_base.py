
import os
import re
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from minisom import MiniSom

warnings.filterwarnings("ignore")

# =========================================================
# НАСТРОЙКИ
# =========================================================
CELL_SIZE = 500
RANDOM_STATE = 42

# методы сохраняем
SOM_X = 12
SOM_Y = 12
SOM_ITERS = 4000
N_CLUSTERS = 6
USE_SUPERVISED = True

# регрессия становится основой,
# но геологический блок и кластеры остаются как стабилизаторы
W_REG = 0.42
W_GEO = 0.34
W_COINCIDENCE = 0.12
W_CLUSTER = 0.05
W_LOCAL = 0.07
SHOW_POINTS = False

# proximity
Q_FACIES = 0.78
Q_PALEO = 0.76
Q_STRUCT = 0.72
Q_MAGM = 0.42
Q_TECT1 = 0.74
Q_TECT2 = 0.74

# визуализация и gold-зоны
N_DISPLAY_CLASSES = 20
TOP_GOLD_Q = 0.05
TOP_LOCAL_Q = 0.95

# =========================================================
# ПУТИ
# =========================================================
def find_existing_base_dir() -> Path:
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
    raise FileNotFoundError("Не найден каталог с shp_dbf. Укажи BASE_DIR вручную.")

BASE_DIR = find_existing_base_dir()
SHP_DIR = BASE_DIR / "shp_dbf"
OUT_DIR = BASE_DIR / "same_methods_fixed_v5_regression_base"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAFE_ALIAS_DIR = OUT_DIR / "_safe_shp_aliases"
SAFE_ALIAS_DIR.mkdir(parents=True, exist_ok=True)

OUT_GPKG = OUT_DIR / "gold_forecast_same_methods_fixed_v5.gpkg"
OUT_PNG = OUT_DIR / "gold_forecast_same_methods_fixed_v5.png"
OUT_PROX = OUT_DIR / "prox_magm_same_methods_fixed_v5.png"
OUT_COMPARE = OUT_DIR / "compare_same_methods_fixed_v5.png"
OUT_CSV = OUT_DIR / "grid_attributes_same_methods_fixed_v5.csv"
OUT_JSON = OUT_DIR / "metrics_same_methods_fixed_v5.json"

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ
# =========================================================
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

def robust_normalize_01(values, q_low=0.03, q_high=0.97):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    lo = np.nanquantile(arr[finite], q_low)
    hi = np.nanquantile(arr[finite], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return normalize_01(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)

def read_sidecar_proj4(shp_path: Path):
    sidecar = shp_path.with_name(shp_path.stem + "_shp.pj4")
    if sidecar.exists():
        txt = sidecar.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"pj4=(.+)", txt)
        if m:
            return m.group(1).strip()
    return None

def prepare_ascii_aliases(shp_dir: Path, alias_dir: Path):
    aliases = {}
    stems = {}
    for name_b in os.listdir(os.fsencode(shp_dir)):
        if not name_b.endswith((b".shp", b".shx", b".dbf", b".prj", b".pj4")):
            continue
        if name_b.endswith(b"_shp.pj4"):
            continue
        base_b, ext_b = os.path.splitext(name_b)
        stems.setdefault(base_b, set()).add(ext_b)

    alias_idx = 0
    for base_b, exts in sorted(stems.items()):
        try:
            base_s = os.fsdecode(base_b)
            safe = all(ord(ch) < 128 and (ch.isalnum() or ch in "_-. ") for ch in base_s)
        except Exception:
            base_s = None
            safe = False

        if safe:
            aliases[base_s] = shp_dir / f"{base_s}.shp"
            continue

        alias_name = f"evidence_{alias_idx:02d}"
        alias_idx += 1
        for ext_b in exts:
            src = os.path.join(os.fsencode(shp_dir), base_b + ext_b)
            dst = alias_dir / f"{alias_name}{os.fsdecode(ext_b)}"
            shutil.copyfile(src, dst)
        pj4_src = os.path.join(os.fsencode(shp_dir), base_b + b"_shp.pj4")
        if os.path.exists(pj4_src):
            shutil.copyfile(pj4_src, alias_dir / f"{alias_name}_shp.pj4")
        aliases[alias_name] = alias_dir / f"{alias_name}.shp"
    return aliases

def load_layer(shp_path: Path):
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        proj4 = read_sidecar_proj4(shp_path)
        if proj4:
            gdf = gdf.set_crs(CRS.from_proj4(proj4), allow_override=True)
    return gdf

def to_crs_safe(gdf, target_crs):
    if gdf.crs is None and target_crs is not None:
        return gdf.set_crs(target_crs, allow_override=True)
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)

def build_grid(mask, cell_size):
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
    grid = gpd.GeoDataFrame(rows, columns=["cell_id", "row", "col", "geometry"], geometry="geometry", crs=mask.crs)
    return grid, mask_union, (len(ys), len(xs))

def add_distance_feature(grid, source, name):
    source_union = unary_union(source.geometry)
    distances = np.empty(len(grid), dtype=float)
    for i, geom in enumerate(grid.geometry.values):
        distances[i] = 0.0 if geom.intersects(source_union) else geom.distance(source_union)
    grid[name] = distances
    return grid

def distance_to_proximity(distance, transform="sqrt", q=0.75):
    d = np.asarray(distance, dtype=float)
    d = np.clip(d, 0, None)
    if transform == "sqrt":
        dt = np.sqrt(d)
    elif transform == "cbrt":
        dt = np.cbrt(d)
    elif transform == "log1p":
        dt = np.log1p(d)
    else:
        dt = d
    scale = float(np.nanquantile(dt, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanmean(dt)), 1.0)
    prox = np.exp(-dt / scale)
    return np.clip(prox, 0, 1)

def smooth_on_regular_grid(grid, value_col, shape, passes=1):
    try:
        from scipy.signal import convolve2d
    except Exception:
        return grid[value_col].to_numpy()
    arr = np.full(shape, np.nan, dtype=float)
    arr[grid["row"].to_numpy(), grid["col"].to_numpy()] = grid[value_col].to_numpy()
    kernel = np.array([[1.0, 1.2, 1.0], [1.2, 3.0, 1.2], [1.0, 1.2, 1.0]], dtype=float)
    smoothed = arr.copy()
    for _ in range(max(1, passes)):
        valid = np.isfinite(smoothed).astype(float)
        filled = np.nan_to_num(smoothed, nan=0.0)
        num = convolve2d(filled, kernel, mode="same", boundary="fill", fillvalue=0)
        den = convolve2d(valid, kernel, mode="same", boundary="fill", fillvalue=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.where(den > 0, num / den, np.nan)
    return smoothed[grid["row"].to_numpy(), grid["col"].to_numpy()]

def collect_points(mask_crs, aliases):
    point_layers = []
    for name, shp_path in aliases.items():
        if name in {"svita_new", "fasii", "glub_raz_nw", "glub_r_nw", "gr_dol_vp_poly", "kory", "dayki_buf"}:
            continue
        gdf = load_layer(shp_path)
        gdf = to_crs_safe(gdf, mask_crs)
        geom_types = {str(x) for x in gdf.geom_type.unique()}
        if "Point" in geom_types or "MultiPoint" in geom_types:
            point_layers.append(gdf)
    if not point_layers:
        return None
    pts = pd.concat(point_layers, ignore_index=True)
    return gpd.GeoDataFrame(pts, geometry="geometry", crs=mask_crs)

def set_mask_extent(ax, mask):
    minx, miny, maxx, maxy = mask.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)


def keep_large_components(grid, bool_col, shape, min_cells=4):
    try:
        from scipy import ndimage
    except Exception:
        return grid[bool_col].to_numpy().astype(bool)
    arr = np.zeros(shape, dtype=np.uint8)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr[rows, cols] = grid[bool_col].to_numpy().astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n = ndimage.label(arr, structure=structure)
    if n == 0:
        return grid[bool_col].to_numpy().astype(bool)
    sizes = np.bincount(labeled.ravel())
    keep = np.isin(labeled, np.where(sizes >= min_cells)[0])
    keep &= labeled > 0
    return keep[rows, cols]

def local_max_mask(grid, value_col, shape):
    try:
        from scipy.ndimage import maximum_filter
    except Exception:
        vals = grid[value_col].to_numpy()
        thr = np.nanquantile(vals, 0.98)
        return vals >= thr
    arr = np.full(shape, np.nan, dtype=float)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    vals = grid[value_col].to_numpy()
    arr[rows, cols] = vals
    filled = np.nan_to_num(arr, nan=-9999.0)
    locmax = maximum_filter(filled, size=3, mode="nearest")
    mask = np.isfinite(arr) & (filled >= locmax)
    return mask[rows, cols]

def make_display_classes(grid):
    prog_disp = robust_normalize_01(grid["prognoz"].to_numpy(), 0.02, 0.98)
    grid["display_score"] = prog_disp
    bins = np.linspace(0, 1, N_DISPLAY_CLASSES + 1)
    grid["display_class"] = np.digitize(prog_disp, bins[1:-1], right=False)
    return grid


def mark_gold_zones(grid, shape, mask_union):
    q_best = float(grid["prognoz"].quantile(TOP_GOLD_Q))
    q_local = float(grid["local_bonus"].quantile(TOP_LOCAL_Q))
    q_coinc = float(grid["coincidence_score"].quantile(TOP_LOCAL_Q))
    q_magm = float(grid["prox_magm"].quantile(0.84))
    q_tmagm = float(grid["tect_magm_intersection"].quantile(0.72))
    q_reg = float(grid["regression_score_sm"].quantile(0.88))

    mask_boundary = mask_union.boundary
    grid["dist_to_boundary"] = np.array([geom.distance(mask_boundary) for geom in grid.geometry])

    local_peak = local_max_mask(grid, "prospectivity", shape)

    seed_gold = (
        (grid["prognoz"] <= q_best) &
        (grid["regression_score_sm"] >= q_reg) &
        (
            (grid["local_bonus"] >= q_local) |
            (grid["coincidence_score"] >= q_coinc) |
            (
                (grid["prox_magm"] >= q_magm) &
                (grid["tect_magm_intersection"] >= q_tmagm)
            )
        )
    )

    edge_gold = (
        (grid["dist_to_boundary"] <= CELL_SIZE * 1.10) &
        (grid["prox_magm"] >= q_magm) &
        (grid["tect_magm_intersection"] >= q_tmagm) &
        (grid["regression_score_sm"] >= float(grid["regression_score_sm"].quantile(0.75)))
    )

    # мягкий контур лучших зон: seeds + локальные максимумы + краевые линейные зоны
    raw_gold = (
        (seed_gold & (local_peak | (grid["coincidence_score"] >= float(grid["coincidence_score"].quantile(0.88))))) |
        edge_gold
    )

    grid["gold_seed"] = raw_gold.astype(int)
    large = keep_large_components(grid, "gold_seed", shape, min_cells=4)
    grid["gold_zone"] = large.astype(int)
    return grid

def plot_prox(grid, mask, out_png):
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.plot(column="prox_magm", ax=ax, cmap="RdYlBu_r", linewidth=0, legend=True)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)
    set_mask_extent(ax, mask)
    ax.set_title("prox_magm")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_final(grid, mask, points, out_png):
    fig, ax = plt.subplots(figsize=(10, 10))
    bins = np.arange(N_DISPLAY_CLASSES + 1)
    norm = BoundaryNorm(bins, plt.cm.bwr.N)
    grid.plot(column="display_class", ax=ax, cmap="bwr", norm=norm, linewidth=0, legend=True)
    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=ax, color="#f2d200", linewidth=0)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)
    if SHOW_POINTS and points is not None and len(points) > 0:
        points.plot(ax=ax, color="yellow", markersize=8, edgecolor="black", linewidth=0.25)
    set_mask_extent(ax, mask)
    ax.set_title("Итоговый прогноз")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_compare(grid, mask, points, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    grid.plot(column="prox_magm", ax=axes[0], cmap="RdYlBu_r", linewidth=0)
    mask.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
    set_mask_extent(axes[0], mask)
    axes[0].set_title("prox_magm")
    axes[0].set_axis_off()

    bins = np.arange(N_DISPLAY_CLASSES + 1)
    norm = BoundaryNorm(bins, plt.cm.bwr.N)
    grid.plot(column="display_class", ax=axes[1], cmap="bwr", norm=norm, linewidth=0)
    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=axes[1], color="#f2d200", linewidth=0)
    mask.boundary.plot(ax=axes[1], color="black", linewidth=0.5)
    if points is not None and len(points) > 0:
        points.plot(ax=axes[1], color="yellow", markersize=8, edgecolor="black", linewidth=0.25)
    set_mask_extent(axes[1], mask)
    axes[1].set_title("Итоговый прогноз")
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# ЗАГРУЗКА
# =========================================================
aliases = prepare_ascii_aliases(SHP_DIR, SAFE_ALIAS_DIR)
mask = load_layer(aliases["svita_new"])
facies = to_crs_safe(load_layer(aliases["fasii"]), mask.crs)
tect1 = to_crs_safe(load_layer(aliases["glub_raz_nw"]), mask.crs)
tect2 = to_crs_safe(load_layer(aliases["glub_r_nw"]), mask.crs)
paleo = to_crs_safe(load_layer(aliases["gr_dol_vp_poly"]), mask.crs)
struct = to_crs_safe(load_layer(aliases["kory"]), mask.crs)
magm = to_crs_safe(load_layer(aliases["dayki_buf"]), mask.crs)
points = collect_points(mask.crs, aliases)

# =========================================================
# СЕТКА
# =========================================================
grid, mask_union, grid_shape = build_grid(mask, CELL_SIZE)

# =========================================================
# ДИСТАНЦИИ
# =========================================================
grid = add_distance_feature(grid, facies, "dist_facies")
grid = add_distance_feature(grid, paleo, "dist_paleo")
grid = add_distance_feature(grid, struct, "dist_struct")
grid = add_distance_feature(grid, magm, "dist_magm")
grid = add_distance_feature(grid, tect1, "dist_tect1")
grid = add_distance_feature(grid, tect2, "dist_tect2")

# =========================================================
# PROXIMITY
# =========================================================
grid["prox_facies"] = distance_to_proximity(grid["dist_facies"], transform="cbrt", q=Q_FACIES)
grid["prox_paleo"] = distance_to_proximity(grid["dist_paleo"], transform="cbrt", q=Q_PALEO)
grid["prox_struct"] = distance_to_proximity(grid["dist_struct"], transform="sqrt", q=Q_STRUCT)
grid["prox_magm"] = distance_to_proximity(grid["dist_magm"], transform="sqrt", q=Q_MAGM)
grid["prox_tect1"] = distance_to_proximity(grid["dist_tect1"], transform="cbrt", q=Q_TECT1)
grid["prox_tect2"] = distance_to_proximity(grid["dist_tect2"], transform="cbrt", q=Q_TECT2)

# =========================================================
# INTERACTIONS
# =========================================================
grid["tect_combo"] = 0.5 * (grid["prox_tect1"] + grid["prox_tect2"])
grid["tect_intersection"] = grid["prox_tect1"] * grid["prox_tect2"]
grid["tect_magm_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_magm"])
grid["tect_struct_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_struct"])
grid["paleo_struct_intersection"] = np.sqrt(grid["prox_paleo"] * grid["prox_struct"])

combo_core = (
    np.clip(grid["tect_combo"], 0, 1) *
    np.clip(0.55 * grid["prox_magm"] + 0.45 * grid["prox_struct"], 0, 1) *
    np.clip(0.60 * grid["prox_paleo"] + 0.40 * grid["prox_facies"], 0, 1)
)
grid["coincidence_score"] = robust_normalize_01(np.sqrt(np.clip(combo_core, 0, 1)), 0.02, 0.98)

tect_support = 0.45 * grid["prox_magm"] + 0.35 * grid["prox_struct"] + 0.20 * grid["prox_paleo"]
grid["tect_only_penalty"] = robust_normalize_01(np.clip(grid["tect_combo"] - tect_support, 0, 1), 0.02, 0.98)

# =========================================================
# GEO SCORE - ослабленный, как стабилизатор
# =========================================================
grid["geo_score_raw"] = (
    0.12 * grid["prox_tect1"] +
    0.12 * grid["prox_tect2"] +
    0.14 * grid["prox_paleo"] +
    0.11 * grid["prox_struct"] +
    0.08 * grid["prox_facies"] +
    0.08 * grid["prox_magm"] +
    0.09 * grid["tect_intersection"] +
    0.09 * grid["tect_magm_intersection"] +
    0.05 * grid["tect_struct_intersection"] +
    0.04 * grid["paleo_struct_intersection"] +
    0.08 * grid["coincidence_score"] -
    0.09 * grid["tect_only_penalty"]
)
grid["geo_score"] = robust_normalize_01(grid["geo_score_raw"], 0.02, 0.98)
grid["geo_score_sm"] = robust_normalize_01(smooth_on_regular_grid(grid, "geo_score", grid_shape, passes=2), 0.02, 0.98)

# =========================================================
# REGRESSION AS BASE
# =========================================================
grid["target"] = 0
grid["regression_score"] = grid["geo_score_sm"]
use_supervised = False
reg_test_auc_proxy = None

feature_cols = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_combo", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "coincidence_score", "tect_only_penalty",
    "geo_score_sm"
]

if USE_SUPERVISED and points is not None and len(points) > 0:
    try:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", op="within")

    positive_cells = joined["cell_id"].dropna().astype(int).unique().tolist()
    grid.loc[grid["cell_id"].isin(positive_cells), "target"] = 1

    pos = int(grid["target"].sum())
    neg = int((grid["target"] == 0).sum())

    if pos >= 20 and neg > pos:
        X = grid[feature_cols].fillna(0).to_numpy()
        y = grid["target"].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Небольшая проверка на holdout, потом учим на всей выборке
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
        lr_eval = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=4000,
            class_weight="balanced",
            C=0.8
        )
        lr_eval.fit(X_train, y_train)
        test_prob = lr_eval.predict_proba(X_test)[:, 1]

        # proxy-метрика без дополнительных библиотек
        y_test = np.asarray(y_test)
        pos_mean = float(np.mean(test_prob[y_test == 1])) if np.any(y_test == 1) else np.nan
        neg_mean = float(np.mean(test_prob[y_test == 0])) if np.any(y_test == 0) else np.nan
        reg_test_auc_proxy = pos_mean - neg_mean

        lr = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=4000,
            class_weight="balanced",
            C=0.8
        )
        lr.fit(X_scaled, y)
        grid["regression_score"] = robust_normalize_01(lr.predict_proba(X_scaled)[:, 1], 0.02, 0.98)
        use_supervised = True

grid["regression_score_sm"] = robust_normalize_01(smooth_on_regular_grid(grid, "regression_score", grid_shape, passes=2), 0.02, 0.98)
grid["ml_score"] = grid["regression_score_sm"]

# =========================================================
# SOM + KMEANS
# =========================================================
X = grid[feature_cols].fillna(0).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

som = MiniSom(
    x=SOM_X, y=SOM_Y, input_len=X_scaled.shape[1],
    sigma=1.1, learning_rate=0.38, random_seed=RANDOM_STATE
)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, SOM_ITERS)

winners = np.array([som.winner(x) for x in X_scaled])
grid["som_x"] = winners[:, 0]
grid["som_y"] = winners[:, 1]
grid["som_node"] = grid["som_x"].astype(str) + "_" + grid["som_y"].astype(str)

som_weights = som.get_weights().reshape(SOM_X * SOM_Y, X_scaled.shape[1])
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
neuron_cluster = kmeans.fit_predict(som_weights)

node_to_cluster = {}
idx = 0
for i in range(SOM_X):
    for j in range(SOM_Y):
        node_to_cluster[f"{i}_{j}"] = int(neuron_cluster[idx])
        idx += 1
grid["cluster"] = grid["som_node"].map(node_to_cluster).astype(int)

cluster_geo = grid.groupby("cluster")["geo_score_sm"].mean().reset_index(name="cluster_geo_mean")
cluster_reg = grid.groupby("cluster")["regression_score"].mean().reset_index(name="cluster_reg_mean")
cluster_coinc = grid.groupby("cluster")["coincidence_score"].mean().reset_index(name="cluster_coinc_mean")
cluster_stats = cluster_geo.merge(cluster_reg, on="cluster", how="outer").merge(cluster_coinc, on="cluster", how="outer")

if use_supervised:
    cluster_hits = grid.groupby("cluster").agg(cells=("cell_id", "count"), positives=("target", "sum")).reset_index()
    cluster_hits["hit_rate"] = cluster_hits["positives"] / cluster_hits["cells"]
    cluster_stats = cluster_stats.merge(cluster_hits, on="cluster", how="left")
    cluster_stats["hit_rate"] = cluster_stats["hit_rate"].fillna(0)
    cluster_stats["cluster_score"] = robust_normalize_01(
        0.35 * cluster_stats["cluster_geo_mean"] +
        0.35 * cluster_stats["cluster_reg_mean"] +
        0.15 * cluster_stats["cluster_coinc_mean"] +
        0.15 * robust_normalize_01(cluster_stats["hit_rate"], 0.0, 1.0),
        0.02, 0.98,
    )
else:
    cluster_stats["cluster_score"] = robust_normalize_01(
        0.45 * cluster_stats["cluster_geo_mean"] +
        0.30 * cluster_stats["cluster_reg_mean"] +
        0.25 * cluster_stats["cluster_coinc_mean"],
        0.02, 0.98,
    )

grid = grid.merge(cluster_stats[["cluster", "cluster_score"]], on="cluster", how="left")
grid["cluster_score"] = grid["cluster_score"].fillna(grid["geo_score_sm"])

# =========================================================
# ИТОГ
# =========================================================
grid["local_bonus"] = robust_normalize_01(
    0.38 * grid["tect_intersection"] +
    0.37 * grid["tect_magm_intersection"] +
    0.25 * grid["tect_struct_intersection"],
    0.02, 0.98,
)

grid["prospectivity_raw"] = (
    W_REG * grid["regression_score_sm"] +
    W_GEO * grid["geo_score_sm"] +
    W_COINCIDENCE * grid["coincidence_score"] +
    W_CLUSTER * grid["cluster_score"] +
    W_LOCAL * grid["local_bonus"]
)

# Дополнительное слабое ослабление tectonic-only после сборки
grid["prospectivity_raw"] = grid["prospectivity_raw"] - 0.06 * grid["tect_only_penalty"]

grid["prospectivity"] = robust_normalize_01(grid["prospectivity_raw"], 0.02, 0.98)

# в логике презентации: меньше = лучше
grid["prognoz"] = 1.0 - grid["prospectivity"]

top_thr = float(grid["prospectivity"].quantile(0.90))
grid["top10"] = (grid["prospectivity"] >= top_thr).astype(int)

try:
    grid["prospect_class"] = pd.qcut(
        grid["prospectivity"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
        duplicates="drop"
    )
except Exception:
    grid["prospect_class"] = "medium"

grid = make_display_classes(grid)
grid = mark_gold_zones(grid, grid_shape, mask_union)

# =========================================================
# СОХРАНЕНИЕ
# =========================================================
if OUT_GPKG.exists():
    OUT_GPKG.unlink()

grid.to_file(OUT_GPKG, layer="forecast_grid", driver="GPKG")
if points is not None and len(points) > 0:
    points.to_file(OUT_GPKG, layer="evidence_points", driver="GPKG")

grid.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

plot_prox(grid, mask, OUT_PROX)
plot_final(grid, mask, points, OUT_PNG)
plot_compare(grid, mask, points, OUT_COMPARE)

metrics = {
    "base_dir": str(BASE_DIR),
    "grid_cells": int(len(grid)),
    "cell_size": CELL_SIZE,
    "use_supervised_requested": bool(USE_SUPERVISED),
    "use_supervised_applied": bool(use_supervised),
    "positive_cells": int(grid["target"].sum()),
    "top10_threshold": float(top_thr),
    "prospectivity_min": float(np.nanmin(grid["prospectivity"])),
    "prospectivity_p05": float(np.nanquantile(grid["prospectivity"], 0.05)),
    "prospectivity_p50": float(np.nanquantile(grid["prospectivity"], 0.50)),
    "prospectivity_p95": float(np.nanquantile(grid["prospectivity"], 0.95)),
    "prospectivity_max": float(np.nanmax(grid["prospectivity"])),
    "prognoz_min": float(np.nanmin(grid["prognoz"])),
    "prognoz_max": float(np.nanmax(grid["prognoz"])),
    "display_score_min": float(np.nanmin(grid["display_score"])),
    "display_score_max": float(np.nanmax(grid["display_score"])),
    "regression_score_min": float(np.nanmin(grid["regression_score"])),
    "regression_score_max": float(np.nanmax(grid["regression_score"])),
    "regression_score_sm_min": float(np.nanmin(grid["regression_score_sm"])),
    "regression_score_sm_max": float(np.nanmax(grid["regression_score_sm"])),
    "reg_test_auc_proxy": None if reg_test_auc_proxy is None else float(reg_test_auc_proxy),
    "gold_zone_count": int(grid["gold_zone"].sum()),
    "point_count": int(len(points)) if points is not None else 0,
}
Path(OUT_JSON).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("Готово.")
print(f"BASE_DIR: {BASE_DIR}")
print(f"PNG: {OUT_PNG}")
print(f"COMPARE: {OUT_COMPARE}")
print(f"GPKG: {OUT_GPKG}")
print(f"CSV: {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
print("Диагностика:")
print(grid[["prospectivity", "prognoz", "display_score", "regression_score", "regression_score_sm", "gold_zone"]].describe())
