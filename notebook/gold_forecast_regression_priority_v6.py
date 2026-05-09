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
from matplotlib.colors import BoundaryNorm, ListedColormap

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

# Регрессия в приоритете, но уже без чрезмерной агрессии
USE_SUPERVISED = True
TEST_SIZE = 0.30

# SOM / KMeans сохраняем как слабую структурную поправку
SOM_X = 12
SOM_Y = 12
SOM_ITERS = 4000
N_CLUSTERS = 6

# Итоговые веса
W_ML = 0.48
W_GEO = 0.24
W_CLUSTER = 0.06
W_COINCIDENCE = 0.12
W_LOCAL_BONUS = 0.10

# proximity
Q_FACIES = 0.78
Q_PALEO = 0.76
Q_STRUCT = 0.72
Q_MAGM = 0.40
Q_TECT1 = 0.74
Q_TECT2 = 0.74

# визуализация
N_DISPLAY_CLASSES = 20
TOP_GOLD_Q = 0.055
TOP_LOCAL_Q = 0.935

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

OUT_DIR = BASE_DIR / "same_methods_regression_priority_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_ALIAS_DIR = OUT_DIR / "_safe_shp_aliases"
SAFE_ALIAS_DIR.mkdir(parents=True, exist_ok=True)

OUT_GPKG = OUT_DIR / "gold_forecast_regression_priority_v6.gpkg"
OUT_PNG = OUT_DIR / "gold_forecast_regression_priority_v6.png"
OUT_COMPARE = OUT_DIR / "compare_regression_priority_v6.png"
OUT_PROX = OUT_DIR / "prox_magm_regression_priority_v6.png"
OUT_CSV = OUT_DIR / "grid_attributes_regression_priority_v6.csv"
OUT_JSON = OUT_DIR / "metrics_regression_priority_v6.json"

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def normalize_01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    mn = np.nanmin(arr[finite])
    mx = np.nanmax(arr[finite])
    out = np.full_like(arr, np.nan, dtype=float)
    if np.isclose(mx, mn):
        out[finite] = 0.5
        return out
    out[finite] = (arr[finite] - mn) / (mx - mn)
    return out

def robust_normalize_01(values, q_low=0.04, q_high=0.96):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    lo = np.nanquantile(arr[finite], q_low)
    hi = np.nanquantile(arr[finite], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return normalize_01(arr)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~finite] = np.nan
    return out

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
            safe = all(ord(ch) < 128 and (ch.isalnum() or ch in "_.- ") for ch in base_s)
        except Exception:
            safe = False
            base_s = None

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

    kernel = np.array(
        [[1.0, 1.25, 1.0],
         [1.25, 3.2, 1.25],
         [1.0, 1.25, 1.0]],
        dtype=float
    )

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

def local_max_mask(grid, value_col, shape):
    try:
        from scipy.ndimage import maximum_filter
    except Exception:
        vals = grid[value_col].to_numpy()
        thr = np.nanquantile(vals, 0.985)
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
    # Компрессия крайних значений, чтобы красный был мягче и меньше резал глаза.
    prog_disp = robust_normalize_01(grid["prognoz"].to_numpy(), 0.06, 0.94)
    prog_disp = 0.5 + 0.72 * (prog_disp - 0.5)
    prog_disp = np.clip(prog_disp, 0, 1)
    grid["display_score"] = prog_disp
    bins = np.linspace(0, 1, N_DISPLAY_CLASSES + 1)
    grid["display_class"] = np.digitize(prog_disp, bins[1:-1], right=False)
    return grid

def mark_gold_zones(grid, shape, mask_union):
    q_best = float(grid["prognoz"].quantile(TOP_GOLD_Q))
    q_local = float(grid["local_bonus"].quantile(TOP_LOCAL_Q))
    q_coinc = float(grid["coincidence_score"].quantile(TOP_LOCAL_Q))
    q_tmagm = float(grid["tect_magm_intersection"].quantile(0.68))
    q_magm = float(grid["prox_magm"].quantile(0.83))

    mask_boundary = mask_union.boundary
    grid["dist_to_boundary"] = np.array([geom.distance(mask_boundary) for geom in grid.geometry])

    local_peak = local_max_mask(grid, "prospectivity_sm", shape)

    core_gold = (
        (grid["prognoz"] <= q_best) &
        local_peak &
        (grid["tect_magm_intersection"] >= q_tmagm) &
        ((grid["local_bonus"] >= q_local) | (grid["coincidence_score"] >= q_coinc))
    )

    # Отдельно спасаем нижние и краевые узкие магмато-тектонические зоны
    edge_gold = (
        (grid["dist_to_boundary"] <= CELL_SIZE * 1.25) &
        (grid["prox_magm"] >= q_magm) &
        (grid["tect_magm_intersection"] >= float(grid["tect_magm_intersection"].quantile(0.70))) &
        (grid["coincidence_score"] >= float(grid["coincidence_score"].quantile(0.70)))
    )

    grid["gold_zone"] = (core_gold | edge_gold).astype(int)
    return grid

def custom_bwr_soft():
    # Более мягкая палитра, чем стандартная bwr
    colors = [
        "#1f1fff", "#3333f0", "#5050ea", "#6f6fe8", "#8c8ce6",
        "#a6a6e6", "#c2c2ea", "#dcdcf0", "#efefef", "#f4eded",
        "#f0d6d6", "#edc0c0", "#eca4a4", "#ec8c8c", "#ee7474",
        "#f25f5f", "#f84d4d", "#ff3d3d", "#ff2b2b", "#ff1e1e"
    ]
    return ListedColormap(colors)

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
    norm = BoundaryNorm(bins, custom_bwr_soft().N)
    grid.plot(column="display_class", ax=ax, cmap=custom_bwr_soft(), norm=norm, linewidth=0, legend=True)
    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=ax, color="#f2d200", linewidth=0)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)
    if points is not None and len(points) > 0:
        points.plot(ax=ax, color="yellow", markersize=7, edgecolor="black", linewidth=0.25)
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
    norm = BoundaryNorm(bins, custom_bwr_soft().N)
    grid.plot(column="display_class", ax=axes[1], cmap=custom_bwr_soft(), norm=norm, linewidth=0)
    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=axes[1], color="#f2d200", linewidth=0)
    mask.boundary.plot(ax=axes[1], color="black", linewidth=0.5)
    if points is not None and len(points) > 0:
        points.plot(ax=axes[1], color="yellow", markersize=7, edgecolor="black", linewidth=0.25)
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
    np.clip(0.58 * grid["prox_magm"] + 0.42 * grid["prox_struct"], 0, 1) *
    np.clip(0.60 * grid["prox_paleo"] + 0.40 * grid["prox_facies"], 0, 1)
)
grid["coincidence_score"] = robust_normalize_01(np.sqrt(np.clip(combo_core, 0, 1)), 0.04, 0.96)

# Штраф для ложных tectonic-only зон усиливаем
tect_support = 0.48 * grid["prox_magm"] + 0.32 * grid["prox_struct"] + 0.20 * grid["prox_paleo"]
grid["tect_only_penalty"] = robust_normalize_01(np.clip(grid["tect_combo"] - tect_support, 0, 1), 0.04, 0.96)

# =========================================================
# GEO PRIOR
# =========================================================
grid["geo_score_raw"] = (
    0.10 * grid["prox_tect1"] +
    0.10 * grid["prox_tect2"] +
    0.14 * grid["prox_paleo"] +
    0.11 * grid["prox_struct"] +
    0.08 * grid["prox_facies"] +
    0.09 * grid["prox_magm"] +
    0.07 * grid["tect_intersection"] +
    0.10 * grid["tect_magm_intersection"] +
    0.05 * grid["tect_struct_intersection"] +
    0.04 * grid["paleo_struct_intersection"] +
    0.11 * grid["coincidence_score"] -
    0.12 * grid["tect_only_penalty"]
)
grid["geo_score"] = robust_normalize_01(grid["geo_score_raw"], 0.04, 0.96)
grid["geo_score_sm"] = robust_normalize_01(smooth_on_regular_grid(grid, "geo_score", grid_shape, passes=1), 0.04, 0.96)

# =========================================================
# TARGET + LOGISTIC REGRESSION
# =========================================================
grid["target"] = 0
grid["ml_score"] = grid["geo_score_sm"]
use_supervised = False
model_metrics = {}

feature_cols = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_combo", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "coincidence_score",
    "tect_only_penalty", "geo_score_sm"
]

if USE_SUPERVISED and points is not None and len(points) > 0:
    try:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", op="within")

    positive_cells = joined["cell_id"].dropna().astype(int).unique().tolist()
    grid.loc[grid["cell_id"].isin(positive_cells), "target"] = 1

    positives = int(grid["target"].sum())
    negatives = int((grid["target"] == 0).sum())

    if positives >= 10 and negatives > positives:
        X = grid[feature_cols].fillna(0).to_numpy()
        y = grid["target"].to_numpy()
        idx = np.arange(len(grid))

        try:
            train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        except Exception:
            train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        scaler_lr = StandardScaler()
        X_train = scaler_lr.fit_transform(X[train_idx])
        X_all = scaler_lr.transform(X)

        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=5000, class_weight="balanced")
        lr.fit(X_train, y[train_idx])

        raw_ml = lr.predict_proba(X_all)[:, 1]
        grid["ml_score_raw"] = raw_ml
        grid["ml_score"] = robust_normalize_01(raw_ml, 0.04, 0.96)

        # Легкое пространственное сглаживание именно regression-score,
        # чтобы убрать слишком резкие красные/синие пятна.
        grid["ml_score_sm"] = robust_normalize_01(
            smooth_on_regular_grid(grid, "ml_score", grid_shape, passes=2),
            0.04, 0.96
        )
        use_supervised = True

        if len(test_idx) > 0 and len(np.unique(y[test_idx])) > 1:
            from sklearn.metrics import roc_auc_score
            model_metrics["test_auc"] = float(roc_auc_score(y[test_idx], raw_ml[test_idx]))

        coef_df = pd.DataFrame({"feature": feature_cols, "coef": lr.coef_[0]}).sort_values("coef", ascending=False)
        coef_df.to_csv(OUT_DIR / "lr_feature_weights_v6.csv", index=False, encoding="utf-8-sig")
        model_metrics["positives"] = positives
        model_metrics["negatives"] = negatives
    else:
        grid["ml_score_sm"] = grid["geo_score_sm"]
else:
    grid["ml_score_sm"] = grid["geo_score_sm"]

# =========================================================
# SOM + KMEANS
# =========================================================
X = grid[feature_cols].fillna(0).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

som = MiniSom(x=SOM_X, y=SOM_Y, input_len=X_scaled.shape[1], sigma=1.1, learning_rate=0.38, random_seed=RANDOM_STATE)
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
cluster_ml = grid.groupby("cluster")["ml_score_sm"].mean().reset_index(name="cluster_ml_mean")
cluster_coinc = grid.groupby("cluster")["coincidence_score"].mean().reset_index(name="cluster_coinc_mean")
cluster_stats = cluster_geo.merge(cluster_ml, on="cluster", how="outer").merge(cluster_coinc, on="cluster", how="outer")

if use_supervised:
    cluster_stats["cluster_score"] = robust_normalize_01(
        0.78 * cluster_stats["cluster_geo_mean"] +
        0.12 * cluster_stats["cluster_ml_mean"] +
        0.10 * cluster_stats["cluster_coinc_mean"],
        0.04, 0.96
    )
else:
    cluster_stats["cluster_score"] = robust_normalize_01(
        0.82 * cluster_stats["cluster_geo_mean"] +
        0.18 * cluster_stats["cluster_coinc_mean"],
        0.04, 0.96
    )

grid = grid.merge(cluster_stats[["cluster", "cluster_score"]], on="cluster", how="left")
grid["cluster_score"] = grid["cluster_score"].fillna(grid["geo_score_sm"])

# =========================================================
# ИТОГ
# =========================================================
grid["local_bonus"] = robust_normalize_01(
    0.30 * grid["tect_intersection"] +
    0.45 * grid["tect_magm_intersection"] +
    0.25 * grid["tect_struct_intersection"],
    0.04, 0.96
)

if use_supervised:
    grid["prospectivity_raw"] = (
        W_ML * grid["ml_score_sm"] +
        W_GEO * grid["geo_score_sm"] +
        W_CLUSTER * grid["cluster_score"] +
        W_COINCIDENCE * grid["coincidence_score"] +
        W_LOCAL_BONUS * grid["local_bonus"]
    )
else:
    grid["prospectivity_raw"] = (
        0.72 * grid["geo_score_sm"] +
        0.06 * grid["cluster_score"] +
        0.12 * grid["coincidence_score"] +
        0.10 * grid["local_bonus"]
    )

grid["prospectivity"] = robust_normalize_01(grid["prospectivity_raw"], 0.04, 0.96)
grid["prospectivity_sm"] = robust_normalize_01(
    smooth_on_regular_grid(grid, "prospectivity", grid_shape, passes=1),
    0.04, 0.96
)

# В логике презентации: меньше prognoz = лучше
grid["prognoz"] = 1.0 - grid["prospectivity_sm"]

top_thr = float(grid["prospectivity_sm"].quantile(0.90))
grid["top10"] = (grid["prospectivity_sm"] >= top_thr).astype(int)

try:
    grid["prospect_class"] = pd.qcut(
        grid["prospectivity_sm"],
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
    "top10_threshold": float(top_thr),
    "point_count": int(len(points)) if points is not None else 0,
    "gold_zone_count": int(grid["gold_zone"].sum()),
    "gold_zone_share": float(grid["gold_zone"].mean()),
    "prospectivity_min": float(np.nanmin(grid["prospectivity_sm"])),
    "prospectivity_p05": float(np.nanquantile(grid["prospectivity_sm"], 0.05)),
    "prospectivity_p50": float(np.nanquantile(grid["prospectivity_sm"], 0.50)),
    "prospectivity_p95": float(np.nanquantile(grid["prospectivity_sm"], 0.95)),
    "prospectivity_max": float(np.nanmax(grid["prospectivity_sm"])),
    "prognoz_min": float(np.nanmin(grid["prognoz"])),
    "prognoz_max": float(np.nanmax(grid["prognoz"])),
    "display_score_min": float(np.nanmin(grid["display_score"])),
    "display_score_max": float(np.nanmax(grid["display_score"])),
    **model_metrics
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
print(grid[["prospectivity_sm", "prognoz", "display_score", "gold_zone"]].describe())
