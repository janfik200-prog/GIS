import os
import re
import glob
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from minisom import MiniSom

warnings.filterwarnings("ignore")

# =========================================================
# НАСТРОЙКИ
# =========================================================
CELL_SIZE = 500
RANDOM_STATE = 42

# SOM / KMeans ОСТАВЛЕНЫ
SOM_X = 12
SOM_Y = 12
SOM_ITERS = 5000
N_CLUSTERS = 6

# LR ОСТАВЛЕНА, НО ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНА,
# чтобы не смешивать прогноз и проверку.
USE_SUPERVISED = False

# Веса итоговой модели
W_GEO = 0.68
W_CLUSTER = 0.22
W_ML = 0.10 if USE_SUPERVISED else 0.0

# Вес локального бонуса — оставляем, но делаем меньше,
# чтобы пересечения не ломали всю карту
LOCAL_BONUS_WEIGHT = 0.08

# Важное: dayki_buf уже буферизованы,
# поэтому для magm делаем более резкий спад
Q_FACIES = 0.75
Q_PALEO = 0.75
Q_STRUCT = 0.75
Q_MAGM = 0.60
Q_TECT1 = 0.75
Q_TECT2 = 0.75

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

    raise FileNotFoundError(
        "Не найден каталог с shp_dbf. Укажи BASE_DIR вручную."
    )

BASE_DIR = find_existing_base_dir()
SHP_DIR = BASE_DIR / "shp_dbf"
OUT_DIR = BASE_DIR / "same_methods_fixed_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_ALIAS_DIR = OUT_DIR / "_safe_shp_aliases"
SAFE_ALIAS_DIR.mkdir(parents=True, exist_ok=True)

OUT_GPKG = OUT_DIR / "gold_forecast_same_methods_fixed.gpkg"
OUT_PNG = OUT_DIR / "gold_forecast_same_methods_fixed.png"
OUT_PROX = OUT_DIR / "prox_magm_same_methods_fixed.png"
OUT_COMPARE = OUT_DIR / "compare_same_methods_fixed.png"
OUT_CSV = OUT_DIR / "grid_attributes_same_methods_fixed.csv"
OUT_JSON = OUT_DIR / "metrics_same_methods_fixed.json"

# =========================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# =========================================================
def normalize_01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mx, mn):
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


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
    if gdf.crs is None and target_crs is None:
        return gdf
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

    grid = gpd.GeoDataFrame(
        rows,
        columns=["cell_id", "row", "col", "geometry"],
        geometry="geometry",
        crs=mask.crs,
    )
    return grid, mask_union, (len(ys), len(xs))


def add_distance_feature(grid, source, name):
    source_union = unary_union(source.geometry)

    # ИСПРАВЛЕНО: расстояние считаем от всей ячейки,
    # а не от центроида. Если пересекает объект — 0.
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
        [[1.0, 1.5, 1.0],
         [1.5, 4.0, 1.5],
         [1.0, 1.5, 1.0]],
        dtype=float,
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
    # Берем только точечные слои как контроль/валидацию
    point_layers = []
    for name, shp_path in aliases.items():
        if name in {
            "svita_new", "fasii", "glub_raz_nw", "glub_r_nw",
            "gr_dol_vp_poly", "kory", "dayki_buf"
        }:
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


def plot_prox(grid, mask, out_png):
    fig, ax = plt.subplots(figsize=(12, 12))
    grid.plot(column="prox_magm", ax=ax, cmap="RdYlBu_r", linewidth=0, legend=True)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title("prox_magm")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_final(grid, mask, points, out_png):
    fig, ax = plt.subplots(figsize=(12, 12))

    # ИСПРАВЛЕНО: визуализация ближе к тому, как ты сравниваешь с эталоном
    grid.plot(column="prognoz", ax=ax, cmap="bwr", linewidth=0, alpha=0.58, legend=True)
    grid[grid["top10"] == 1].plot(ax=ax, color="yellow", linewidth=0, alpha=0.92)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)

    if points is not None and len(points) > 0:
        points.plot(ax=ax, color="yellow", markersize=14, edgecolor="black", linewidth=0.35)

    ax.set_title("Итоговый прогноз")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compare(grid, mask, points, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    grid.plot(column="prox_magm", ax=axes[0], cmap="RdYlBu_r", linewidth=0)
    mask.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
    axes[0].set_title("prox_magm")
    axes[0].set_axis_off()

    grid.plot(column="prognoz", ax=axes[1], cmap="bwr", linewidth=0, alpha=0.58)
    grid[grid["top10"] == 1].plot(ax=axes[1], color="yellow", linewidth=0, alpha=0.92)
    mask.boundary.plot(ax=axes[1], color="black", linewidth=0.5)
    if points is not None and len(points) > 0:
        points.plot(ax=axes[1], color="yellow", markersize=12, edgecolor="black", linewidth=0.35)
    axes[1].set_title("Итоговый прогноз")
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# ЗАГРУЗКА ДАННЫХ
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
# ПРИЗНАКИ: ДИСТАНЦИИ
# =========================================================
grid = add_distance_feature(grid, facies, "dist_facies")
grid = add_distance_feature(grid, paleo, "dist_paleo")
grid = add_distance_feature(grid, struct, "dist_struct")
grid = add_distance_feature(grid, magm, "dist_magm")
grid = add_distance_feature(grid, tect1, "dist_tect1")
grid = add_distance_feature(grid, tect2, "dist_tect2")

# =========================================================
# ПРИЗНАКИ: PROXIMITY
# =========================================================
grid["prox_facies"] = distance_to_proximity(grid["dist_facies"], transform="cbrt", q=Q_FACIES)
grid["prox_paleo"] = distance_to_proximity(grid["dist_paleo"], transform="cbrt", q=Q_PALEO)
grid["prox_struct"] = distance_to_proximity(grid["dist_struct"], transform="sqrt", q=Q_STRUCT)
grid["prox_magm"] = distance_to_proximity(grid["dist_magm"], transform="sqrt", q=Q_MAGM)
grid["prox_tect1"] = distance_to_proximity(grid["dist_tect1"], transform="cbrt", q=Q_TECT1)
grid["prox_tect2"] = distance_to_proximity(grid["dist_tect2"], transform="cbrt", q=Q_TECT2)

# =========================================================
# INTERACTIONS — ОСТАВЛЕНЫ, НО СДЕЛАНЫ СТАБИЛЬНЕЕ
# =========================================================
grid["tect_combo"] = 0.5 * (grid["prox_tect1"] + grid["prox_tect2"])
grid["tect_intersection"] = grid["prox_tect1"] * grid["prox_tect2"]

# ИСПРАВЛЕНО: вместо грубого усиления берем геометрические средние,
# чтобы случайное совпадение факторов не перегибало результат.
grid["tect_magm_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_magm"])
grid["tect_struct_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_struct"])
grid["paleo_struct_intersection"] = np.sqrt(grid["prox_paleo"] * grid["prox_struct"])

# =========================================================
# GEO SCORE — ОСТАВЛЕН, НО ВЕСА СБАЛАНСИРОВАНЫ
# =========================================================
grid["geo_score_raw"] = (
    0.20 * grid["prox_tect1"] +
    0.20 * grid["prox_tect2"] +
    0.14 * grid["prox_paleo"] +
    0.13 * grid["prox_struct"] +
    0.10 * grid["prox_facies"] +
    0.09 * grid["prox_magm"] +
    0.06 * grid["tect_intersection"] +
    0.04 * grid["tect_magm_intersection"] +
    0.04 * grid["tect_struct_intersection"]
)

# ИСПРАВЛЕНО: сначала считаем геологический score,
# потом мягко сглаживаем уже итог, а не исходные слои.
grid["geo_score"] = normalize_01(grid["geo_score_raw"])
grid["geo_score_sm"] = normalize_01(smooth_on_regular_grid(grid, "geo_score", grid_shape, passes=2))

# =========================================================
# TARGET / LR — МЕТОД ОСТАВЛЕН, НО НЕ ЛОМАЕТ БАЗОВУЮ ЛОГИКУ
# =========================================================
grid["target"] = 0
grid["ml_score"] = grid["geo_score_sm"]
use_supervised = False

feature_cols = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_combo", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "geo_score_sm"
]

if USE_SUPERVISED and points is not None and len(points) > 0:
    try:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", op="within")

    positive_cells = joined["cell_id"].dropna().astype(int).unique().tolist()
    grid.loc[grid["cell_id"].isin(positive_cells), "target"] = 1

    if grid["target"].sum() >= 10 and grid["target"].sum() < len(grid):
        X = grid[feature_cols].fillna(0).to_numpy()
        y = grid["target"].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=4000,
            class_weight="balanced"
        )
        lr.fit(X_scaled, y)
        grid["ml_score"] = normalize_01(lr.predict_proba(X_scaled)[:, 1])
        use_supervised = True

# =========================================================
# SOM + KMEANS — ОСТАВЛЕНЫ
# =========================================================
X = grid[feature_cols].fillna(0).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

som = MiniSom(
    x=SOM_X,
    y=SOM_Y,
    input_len=X_scaled.shape[1],
    sigma=1.2,
    learning_rate=0.40,
    random_seed=RANDOM_STATE,
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

# =========================================================
# CLUSTER SCORE — ОСТАВЛЕН, НО ОСЛАБЛЕН
# =========================================================
cluster_geo = grid.groupby("cluster")["geo_score_sm"].mean().reset_index(name="cluster_geo_mean")
cluster_ml = grid.groupby("cluster")["ml_score"].mean().reset_index(name="cluster_ml_mean")
cluster_stats = cluster_geo.merge(cluster_ml, on="cluster", how="outer")

if use_supervised:
    cluster_hits = (
        grid.groupby("cluster")
        .agg(cells=("cell_id", "count"), positives=("target", "sum"))
        .reset_index()
    )
    cluster_hits["hit_rate"] = cluster_hits["positives"] / cluster_hits["cells"]
    cluster_stats = cluster_stats.merge(cluster_hits, on="cluster", how="left")
    cluster_stats["hit_rate"] = cluster_stats["hit_rate"].fillna(0)
    cluster_stats["cluster_score"] = normalize_01(
        0.55 * cluster_stats["cluster_geo_mean"] +
        0.25 * cluster_stats["cluster_ml_mean"] +
        0.20 * normalize_01(cluster_stats["hit_rate"])
    )
else:
    cluster_stats["cluster_score"] = normalize_01(
        0.75 * cluster_stats["cluster_geo_mean"] +
        0.25 * cluster_stats["cluster_ml_mean"]
    )

grid = grid.merge(cluster_stats[["cluster", "cluster_score"]], on="cluster", how="left")
grid["cluster_score"] = grid["cluster_score"].fillna(grid["geo_score_sm"])

# =========================================================
# ИТОГ
# =========================================================
if use_supervised:
    grid["prospectivity_raw"] = (
        0.58 * grid["geo_score_sm"] +
        0.20 * grid["cluster_score"] +
        0.22 * grid["ml_score"]
    )
else:
    # ИСПРАВЛЕНО: кластеры не делают половину результата,
    # а только слегка стабилизируют геологическую картину.
    grid["prospectivity_raw"] = (
        W_GEO * grid["geo_score_sm"] +
        W_CLUSTER * grid["cluster_score"]
    )

# Локальный бонус оставлен, но не должен всё перетаскивать на себя
grid["local_bonus"] = normalize_01(
    0.40 * grid["tect_intersection"] +
    0.35 * grid["tect_magm_intersection"] +
    0.25 * grid["tect_struct_intersection"]
)

grid["prospectivity_raw"] += LOCAL_BONUS_WEIGHT * grid["local_bonus"]
grid["prospectivity"] = normalize_01(grid["prospectivity_raw"])

# Для сопоставления с эталоном: меньше = лучше
grid["prognoz"] = 1.0 - grid["prospectivity"]

# ИСПРАВЛЕНО: top зоны берем по 90 перцентилю, а не по qcut-классу
threshold = float(grid["prospectivity"].quantile(0.90))
grid["top10"] = (grid["prospectivity"] >= threshold).astype(int)

try:
    grid["prospect_class"] = pd.qcut(
        grid["prospectivity"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
        duplicates="drop"
    )
except Exception:
    grid["prospect_class"] = "medium"

# =========================================================
# СОХРАНЕНИЕ И МЕТРИКИ
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
    "top10_threshold": threshold,
    "mean_prospectivity": float(grid["prospectivity"].mean()),
    "max_prospectivity": float(grid["prospectivity"].max()),
    "point_count": int(len(points)) if points is not None else 0,
}

Path(OUT_JSON).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("Готово.")
print(f"BASE_DIR: {BASE_DIR}")
print(f"PNG: {OUT_PNG}")
print(f"GPKG: {OUT_GPKG}")
print(f"CSV: {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
