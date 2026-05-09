# ============================================================
# v12 CLEAN: прогноз перспективных зон без SOM и линейной регрессии
# Основа: геологический скоринг + Random Forest
# ============================================================

import json
import os
import re
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================
# SETTINGS
# =========================
CELL_SIZE = 500
RANDOM_STATE = 42
USE_SUPERVISED = True

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 4
RF_MIN_SAMPLES_SPLIT = 8

# Главная формула итогового прогноза
W_RF = 0.65
W_GEO = 0.35

# Настройки превращения расстояний в proximity
Q_FACIES = 0.78
Q_PALEO = 0.76
Q_STRUCT = 0.72
Q_MAGM = 0.42
Q_TECT1 = 0.74
Q_TECT2 = 0.74

N_DISPLAY_CLASSES = 20
SHOW_POINTS = False

# =========================
# PATHS
# =========================
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
    raise FileNotFoundError("Не найден каталог shp_dbf со слоем svita_new.shp")

BASE_DIR = find_base_dir()
SHP_DIR = BASE_DIR / "shp_dbf"
OUT_DIR = BASE_DIR / "rf_clean_v12"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAFE_ALIAS_DIR = OUT_DIR / "_safe_aliases"
SAFE_ALIAS_DIR.mkdir(parents=True, exist_ok=True)

OUT_GPKG = OUT_DIR / "forecast_rf_v12.gpkg"
OUT_PNG = OUT_DIR / "forecast_rf_v12.png"
OUT_COMPARE = OUT_DIR / "compare_rf_v12.png"
OUT_PROX = OUT_DIR / "prox_magm_v12.png"
OUT_CSV = OUT_DIR / "grid_rf_v12.csv"
OUT_JSON = OUT_DIR / "metrics_rf_v12.json"

# =========================
# HELPERS
# =========================
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
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return normalize_01(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def smooth_on_regular_grid(grid, value_col, shape, passes=1):
    """Лёгкое сглаживание только для карты, не для ручной подгонки."""
    try:
        from scipy.signal import convolve2d
    except Exception:
        return grid[value_col].to_numpy()

    arr = np.full(shape, np.nan, dtype=float)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr[rows, cols] = grid[value_col].to_numpy()

    kernel = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=float)

    smoothed = arr.copy()
    for _ in range(max(1, passes)):
        valid = np.isfinite(smoothed).astype(float)
        filled = np.nan_to_num(smoothed, nan=0.0)
        num = convolve2d(filled, kernel, mode="same", boundary="fill", fillvalue=0)
        den = convolve2d(valid, kernel, mode="same", boundary="fill", fillvalue=0)
        smoothed = np.where(den > 0, num / den, np.nan)

    return smoothed[rows, cols]


def keep_large_components(grid, bool_col, shape, min_cells=4):
    """Оставляет только связные перспективные зоны, чтобы убрать одиночный шум."""
    try:
        from scipy import ndimage
    except Exception:
        return grid[bool_col].to_numpy().astype(bool)

    arr = np.zeros(shape, dtype=np.uint8)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr[rows, cols] = grid[bool_col].to_numpy().astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, _ = ndimage.label(arr, structure=structure)
    sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(sizes >= min_cells)[0]
    keep = np.isin(labeled, keep_labels) & (labeled > 0)
    return keep[rows, cols]


def read_sidecar_proj4(shp_path: Path):
    sidecar = shp_path.with_name(shp_path.stem + "_shp.pj4")
    if sidecar.exists():
        txt = sidecar.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"pj4=(.+)", txt)
        if m:
            return m.group(1).strip()
    return None


def prepare_ascii_aliases(shp_dir: Path, alias_dir: Path):
    """Копирует слои с нечитаемыми именами в ascii-алиасы, если надо."""
    aliases, stems = {}, {}

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

        pj4_src = os.path.join(os.fsencode(shp_dir), base_b + b"_shp.pj4")
        if os.path.exists(pj4_src):
            shutil.copyfile(pj4_src, alias_dir / f"{alias}_shp.pj4")

        aliases[alias] = alias_dir / f"{alias}.shp"

    return aliases


def load_layer(path: Path):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.crs is None:
        proj4 = read_sidecar_proj4(path)
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

    grid = gpd.GeoDataFrame(
        rows,
        columns=["cell_id", "row", "col", "geometry"],
        geometry="geometry",
        crs=mask.crs,
    )
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
        t = np.sqrt(d)
    elif transform == "cbrt":
        t = np.cbrt(d)
    else:
        t = d

    scale = float(np.nanquantile(t, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanmean(t)), 1.0)

    return np.clip(np.exp(-t / scale), 0, 1)


def collect_points(mask_crs, aliases):
    base_layers = {
        "svita_new", "fasii", "glub_raz_nw", "glub_r_nw",
        "gr_dol_vp_poly", "kory", "dayki_buf"
    }
    point_layers = []

    for name, shp_path in aliases.items():
        if name in base_layers:
            continue
        gdf = to_crs_safe(load_layer(shp_path), mask_crs)
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


def make_display_classes(grid):
    grid["display_score"] = robust_normalize_01(grid["prospectivity"], 0.02, 0.98)
    bins = np.linspace(0, 1, N_DISPLAY_CLASSES + 1)
    grid["display_class"] = np.digitize(grid["display_score"], bins[1:-1], right=False)
    return grid


def mark_gold_zones_simple(grid, shape):
    """
    Простая логика зон:
    зона перспективна, если входит в верхние 10% по итоговой перспективности.
    Потом убираются одиночные шумовые клетки.
    """
    threshold = float(grid["prospectivity"].quantile(0.90))
    grid["gold_seed"] = (grid["prospectivity"] >= threshold).astype(int)
    grid["gold_zone"] = keep_large_components(grid, "gold_seed", shape, min_cells=4).astype(int)
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
    norm = BoundaryNorm(bins, plt.cm.bwr_r.N)
    grid.plot(column="display_class", ax=ax, cmap="bwr_r", norm=norm, linewidth=0, legend=True)

    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=ax, color="#f2d200", linewidth=0)

    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)

    if SHOW_POINTS and points is not None and len(points) > 0:
        points.plot(ax=ax, color="yellow", markersize=8, edgecolor="black", linewidth=0.25)

    ax.legend(handles=[Patch(facecolor="#f2d200", edgecolor="black", label="Gold zones")], loc="lower right")
    set_mask_extent(ax, mask)
    ax.set_title("Итоговый прогноз v12: RF + GEO")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compare(grid, mask, points, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    grid.plot(column="prox_magm", ax=axes[0], cmap="RdYlBu_r", linewidth=0, legend=True)
    mask.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
    set_mask_extent(axes[0], mask)
    axes[0].set_title("prox_magm")
    axes[0].set_axis_off()

    bins = np.arange(N_DISPLAY_CLASSES + 1)
    norm = BoundaryNorm(bins, plt.cm.bwr_r.N)
    grid.plot(column="display_class", ax=axes[1], cmap="bwr_r", norm=norm, linewidth=0, legend=True)

    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=axes[1], color="#f2d200", linewidth=0)

    mask.boundary.plot(ax=axes[1], color="black", linewidth=0.5)

    if SHOW_POINTS and points is not None and len(points) > 0:
        points.plot(ax=axes[1], color="yellow", markersize=8, edgecolor="black", linewidth=0.25)

    axes[1].legend(handles=[Patch(facecolor="#f2d200", edgecolor="black", label="Gold zones")], loc="lower right")
    set_mask_extent(axes[1], mask)
    axes[1].set_title("Итоговый прогноз v12")
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# LOAD DATA
# =========================
aliases = prepare_ascii_aliases(SHP_DIR, SAFE_ALIAS_DIR)

mask = load_layer(aliases["svita_new"])
facies = to_crs_safe(load_layer(aliases["fasii"]), mask.crs)
tect1 = to_crs_safe(load_layer(aliases["glub_raz_nw"]), mask.crs)
tect2 = to_crs_safe(load_layer(aliases["glub_r_nw"]), mask.crs)
paleo = to_crs_safe(load_layer(aliases["gr_dol_vp_poly"]), mask.crs)
struct = to_crs_safe(load_layer(aliases["kory"]), mask.crs)
magm = to_crs_safe(load_layer(aliases["dayki_buf"]), mask.crs)
points = collect_points(mask.crs, aliases)

# =========================
# GRID + FEATURES
# =========================
grid, mask_union, grid_shape = build_grid(mask, CELL_SIZE)

for src, name in [
    (facies, "dist_facies"),
    (paleo, "dist_paleo"),
    (struct, "dist_struct"),
    (magm, "dist_magm"),
    (tect1, "dist_tect1"),
    (tect2, "dist_tect2"),
]:
    grid = add_distance_feature(grid, src, name)

# Близости к геологическим факторам
grid["prox_facies"] = distance_to_proximity(grid["dist_facies"], "cbrt", Q_FACIES)
grid["prox_paleo"] = distance_to_proximity(grid["dist_paleo"], "cbrt", Q_PALEO)
grid["prox_struct"] = distance_to_proximity(grid["dist_struct"], "sqrt", Q_STRUCT)
grid["prox_magm"] = distance_to_proximity(grid["dist_magm"], "sqrt", Q_MAGM)
grid["prox_tect1"] = distance_to_proximity(grid["dist_tect1"], "cbrt", Q_TECT1)
grid["prox_tect2"] = distance_to_proximity(grid["dist_tect2"], "cbrt", Q_TECT2)

# Геологические взаимодействия
grid["tect_combo"] = 0.5 * (grid["prox_tect1"] + grid["prox_tect2"])
grid["tect_intersection"] = grid["prox_tect1"] * grid["prox_tect2"]
grid["tect_magm_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_magm"])
grid["tect_struct_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_struct"])
grid["paleo_struct_intersection"] = np.sqrt(grid["prox_paleo"] * grid["prox_struct"])

combo_core = (
    np.clip(grid["tect_combo"], 0, 1)
    * np.clip(0.55 * grid["prox_magm"] + 0.45 * grid["prox_struct"], 0, 1)
    * np.clip(0.60 * grid["prox_paleo"] + 0.40 * grid["prox_facies"], 0, 1)
)
grid["coincidence_score"] = robust_normalize_01(np.sqrt(np.clip(combo_core, 0, 1)), 0.02, 0.98)

# =========================
# GEO SCORE
# =========================
grid["geo_score_raw"] = (
    0.14 * grid["prox_tect1"] +
    0.14 * grid["prox_tect2"] +
    0.13 * grid["prox_paleo"] +
    0.11 * grid["prox_struct"] +
    0.09 * grid["prox_magm"] +
    0.07 * grid["prox_facies"] +
    0.08 * grid["tect_intersection"] +
    0.09 * grid["tect_magm_intersection"] +
    0.06 * grid["tect_struct_intersection"] +
    0.04 * grid["paleo_struct_intersection"] +
    0.05 * grid["coincidence_score"]
)
grid["geo_score"] = robust_normalize_01(grid["geo_score_raw"], 0.02, 0.98)
grid["geo_score_sm"] = robust_normalize_01(smooth_on_regular_grid(grid, "geo_score", grid_shape, passes=1), 0.02, 0.98)

# =========================
# RANDOM FOREST
# =========================
grid["target"] = 0
grid["rf_score"] = grid["geo_score_sm"]
feature_importance = {}
rf_test_proxy = None
use_supervised = False

feature_cols = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_combo", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "coincidence_score", "geo_score_sm",
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        rf_eval = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_eval.fit(X_train, y_train)
        test_prob = rf_eval.predict_proba(X_test)[:, 1]
        pos_mean = float(np.mean(test_prob[y_test == 1])) if np.any(y_test == 1) else np.nan
        neg_mean = float(np.mean(test_prob[y_test == 0])) if np.any(y_test == 0) else np.nan
        rf_test_proxy = pos_mean - neg_mean

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X, y)
        grid["rf_score"] = robust_normalize_01(rf.predict_proba(X)[:, 1], 0.02, 0.98)
        feature_importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))
        use_supervised = True
    else:
        print(f"Недостаточно положительных ячеек для RF: positives={pos}. Используется geo_score.")
else:
    print("Точки подтверждения не найдены. Используется geo_score без RF-обучения.")

grid["rf_score_sm"] = robust_normalize_01(smooth_on_regular_grid(grid, "rf_score", grid_shape, passes=1), 0.02, 0.98)

# =========================
# FINAL SURFACE
# =========================
if use_supervised:
    grid["prospectivity_raw"] = W_RF * grid["rf_score_sm"] + W_GEO * grid["geo_score_sm"]
else:
    grid["prospectivity_raw"] = grid["geo_score_sm"]

grid["prospectivity"] = robust_normalize_01(grid["prospectivity_raw"], 0.02, 0.98)
grid["prognoz"] = 1.0 - grid["prospectivity"]

grid = make_display_classes(grid)
grid = mark_gold_zones_simple(grid, grid_shape)

# =========================
# SAVE
# =========================
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
    "version": "v12_clean_rf_geo",
    "base_dir": str(BASE_DIR),
    "grid_cells": int(len(grid)),
    "cell_size": CELL_SIZE,
    "use_supervised_requested": bool(USE_SUPERVISED),
    "use_supervised_applied": bool(use_supervised),
    "som_used": False,
    "linear_regression_used": False,
    "edge_logic_used": False,
    "local_bonus_used": False,
    "penalty_used": False,
    "positive_cells": int(grid["target"].sum()),
    "point_count": int(len(points)) if points is not None else 0,
    "weights": {"rf": W_RF if use_supervised else 0.0, "geo": W_GEO if use_supervised else 1.0},
    "prospectivity_min": float(np.nanmin(grid["prospectivity"])),
    "prospectivity_p50": float(np.nanquantile(grid["prospectivity"], 0.50)),
    "prospectivity_p90": float(np.nanquantile(grid["prospectivity"], 0.90)),
    "prospectivity_p95": float(np.nanquantile(grid["prospectivity"], 0.95)),
    "prospectivity_max": float(np.nanmax(grid["prospectivity"])),
    "gold_zone_count": int(grid["gold_zone"].sum()),
    "gold_zone_share": float(grid["gold_zone"].mean()),
    "rf_test_proxy": None if rf_test_proxy is None else float(rf_test_proxy),
    "rf_feature_importance": feature_importance,
}
OUT_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("Готово: v12_clean_rf_geo")
print(f"PNG: {OUT_PNG}")
print(f"COMPARE: {OUT_COMPARE}")
print(f"GPKG: {OUT_GPKG}")
print(f"CSV: {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
print(grid[["prospectivity", "prognoz", "rf_score_sm", "geo_score_sm", "gold_zone"]].describe())
