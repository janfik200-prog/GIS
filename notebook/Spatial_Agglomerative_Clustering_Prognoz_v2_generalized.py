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
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# ПАРАМЕТРЫ
# =========================
CELL_SIZE = 500
RANDOM_STATE = 42

# Пространственная кластеризация
AUTO_SELECT_CLUSTERS = False
CLUSTER_CANDIDATES = [5, 6, 7, 8, 9]
N_CLUSTERS_FIXED = 7
PCA_COMPONENTS = 5

# Постобработка и отображение
SMOOTH_PASSES = 3
N_DISPLAY_CLASSES = 20
SHOW_POINTS = False

# Выделение top-зон: относительно строгий, но переносимый режим
TOP_ZONE_CORE_Q = 0.992
TOP_ZONE_SIGNAL_Q = 0.95
TOP_ZONE_CLUSTER_PCT_Q = 0.92
LOCAL_PEAK_SIZE = 5
MIN_TOP_ZONE_FRACTION = 0.0004  # минимальный размер пятна как доля от числа ячеек


# =========================
# СЛУЖЕБНЫЕ ФУНКЦИИ
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
OUT_DIR = BASE_DIR / "spatial_agglomerative_result_v2"
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
            d[i] = geom.distance(source_union)
    grid[name] = d
    return grid


def distance_to_proximity(distance, transform="sqrt", q=0.75):
    d = np.asarray(distance, dtype=float)
    d = np.clip(d, 0, None)

    if transform == "sqrt":
        t = np.sqrt(d)
    elif transform == "cbrt":
        t = np.cbrt(d)
    elif transform == "log1p":
        t = np.log1p(d)
    else:
        t = d

    scale = float(np.nanquantile(t, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanmean(t)), 1.0)

    return np.clip(np.exp(-t / scale), 0, 1)


def smooth_on_regular_grid(grid: gpd.GeoDataFrame, values, shape, passes=1):
    from scipy.signal import convolve2d

    arr = np.full(shape, np.nan, dtype=float)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr[rows, cols] = np.asarray(values, dtype=float)

    kernel = np.array([
        [1.0, 1.2, 1.0],
        [1.2, 3.0, 1.2],
        [1.0, 1.2, 1.0],
    ], dtype=float)

    smoothed = arr.copy()
    for _ in range(max(1, passes)):
        valid = np.isfinite(smoothed).astype(float)
        filled = np.nan_to_num(smoothed, nan=0.0)
        num = convolve2d(filled, kernel, mode="same", boundary="fill", fillvalue=0)
        den = convolve2d(valid, kernel, mode="same", boundary="fill", fillvalue=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.divide(
                num,
                den,
                out=np.full_like(num, np.nan, dtype=float),
                where=den > 0,
            )

    return smoothed[rows, cols]


def local_max_mask(grid: gpd.GeoDataFrame, value_col: str, shape, size=3):
    from scipy.ndimage import maximum_filter

    arr = np.full(shape, np.nan, dtype=float)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    vals = grid[value_col].to_numpy(dtype=float)
    arr[rows, cols] = vals

    filled = np.nan_to_num(arr, nan=-9999.0)
    locmax = maximum_filter(filled, size=size, mode="nearest")
    return (np.isfinite(arr) & (filled >= locmax))[rows, cols]


def connected_component_filter(grid: gpd.GeoDataFrame, mask_col: str, shape, min_cells=2):
    from scipy.ndimage import label

    arr = np.zeros(shape, dtype=bool)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr[rows, cols] = grid[mask_col].to_numpy().astype(bool)

    structure = np.ones((3, 3), dtype=int)
    labels, n = label(arr, structure=structure)
    if n == 0:
        return grid[mask_col].to_numpy().astype(bool)

    counts = np.bincount(labels.ravel())
    keep = counts >= min_cells
    keep[0] = False
    return keep[labels][rows, cols]


def make_display_classes(grid: gpd.GeoDataFrame):
    disp = robust_normalize_01(grid["prospectivity"].to_numpy(), 0.02, 0.98)
    grid["display_score"] = disp
    bins = np.linspace(0, 1, N_DISPLAY_CLASSES + 1)
    grid["display_class"] = np.digitize(disp, bins[1:-1], right=False)
    return grid


def mark_top_zones(grid: gpd.GeoDataFrame, shape):
    min_cells = max(2, int(round(len(grid) * MIN_TOP_ZONE_FRACTION)))

    q_core = float(grid["prospectivity"].quantile(TOP_ZONE_CORE_Q))
    q_signal = float(grid["local_strength"].quantile(TOP_ZONE_SIGNAL_Q))
    q_pct = float(grid["cluster_percentile"].quantile(TOP_ZONE_CLUSTER_PCT_Q))
    local_peak = local_max_mask(grid, "prospectivity", shape, size=LOCAL_PEAK_SIZE)

    grid["top_zone"] = (
        (grid["prospectivity"] >= q_core) &
        (grid["local_strength"] >= q_signal) &
        (grid["cluster_percentile"] >= q_pct) &
        local_peak
    ).astype(int)

    grid["top_zone"] = connected_component_filter(grid, "top_zone", shape, min_cells=min_cells).astype(int)

    # Осторожный fallback, если фильтр оказался слишком строгим
    if int(grid["top_zone"].sum()) == 0:
        q_core_fb = float(grid["prospectivity"].quantile(0.99))
        q_signal_fb = float(grid["local_strength"].quantile(0.93))
        local_peak_fb = local_max_mask(grid, "prospectivity", shape, size=max(3, LOCAL_PEAK_SIZE - 2))
        grid["top_zone"] = (
            (grid["prospectivity"] >= q_core_fb) &
            (grid["local_strength"] >= q_signal_fb) &
            local_peak_fb
        ).astype(int)
        grid["top_zone"] = connected_component_filter(grid, "top_zone", shape, min_cells=2).astype(int)

    return grid


def set_mask_extent(ax, mask: gpd.GeoDataFrame):
    minx, miny, maxx, maxy = mask.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)


def plot_final(grid: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, points, out_png: Path):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    bins = np.arange(N_DISPLAY_CLASSES + 1)
    norm = BoundaryNorm(bins, plt.cm.bwr_r.N)
    grid.plot(column="display_class", ax=ax, cmap="bwr_r", norm=norm, linewidth=0, legend=False)

    top = grid[grid["top_zone"] == 1]
    if len(top) > 0:
        top.plot(ax=ax, color="#f2d200", linewidth=0)

    mask.boundary.plot(ax=ax, color="white", linewidth=0.3)

    if SHOW_POINTS and points is not None and len(points) > 0:
        points.plot(ax=ax, color="black", markersize=10, edgecolor="white", linewidth=0.25)

    ax.legend(
        handles=[Patch(facecolor="#f2d200", edgecolor="black", label="Top zone")],
        loc="lower right",
        frameon=True,
    )

    set_mask_extent(ax, mask)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def collect_points(mask_crs, aliases):
    base_names = {
        "svita_new", "fasii", "glub_raz_nw", "glub_r_nw",
        "gr_dol_vp_poly", "kory", "dayki_buf"
    }

    layers = []
    info = []

    for name, shp_path in aliases.items():
        if name in base_names:
            continue
        try:
            gdf = load_layer(shp_path)
        except Exception as exc:
            info.append({"layer": name, "error": str(exc)})
            continue

        geom_types = sorted(set(gdf.geometry.geom_type.astype(str))) if len(gdf) else []
        info.append({"layer": name, "n": int(len(gdf)), "geom_types": geom_types})

        if len(gdf) and set(geom_types).issubset({"Point", "MultiPoint"}):
            gdf = to_crs_safe(gdf, mask_crs)
            gdf = gdf[["geometry"]].copy()
            gdf["source_layer"] = name
            layers.append(gdf)

    if not layers:
        return None, info

    pts = pd.concat(layers, ignore_index=True)
    return gpd.GeoDataFrame(pts, geometry="geometry", crs=mask_crs), info


def choose_cluster_count(X_pca, connectivity, candidates, random_state=42):
    diagnostics = []
    rs = np.random.RandomState(random_state)
    sample_size = min(6000, len(X_pca))
    sample_idx = rs.choice(len(X_pca), size=sample_size, replace=False)
    X_sample = X_pca[sample_idx]

    best_score = -np.inf
    best_k = candidates[0]
    best_labels = None

    for k in candidates:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            connectivity=connectivity,
        )
        labels = model.fit_predict(X_pca)
        sizes = np.bincount(labels)

        try:
            sil = float(silhouette_score(X_sample, labels[sample_idx])) if len(np.unique(labels[sample_idx])) > 1 else np.nan
        except Exception:
            sil = np.nan

        balance = float(np.min(sizes) / np.mean(sizes)) if len(sizes) else 0.0
        entropy = float(-(sizes / sizes.sum() * np.log((sizes / sizes.sum()) + 1e-12)).sum() / np.log(len(sizes))) if len(sizes) > 1 else 0.0
        combined = (0.75 * sil if np.isfinite(sil) else -1.0) + 0.15 * balance + 0.10 * entropy

        diagnostics.append({
            "k": int(k),
            "silhouette": sil,
            "balance": balance,
            "entropy": entropy,
            "combined_score": combined,
            "cluster_sizes": sizes.tolist(),
        })

        if combined > best_score:
            best_score = combined
            best_k = k
            best_labels = labels

    return best_k, best_labels, pd.DataFrame(diagnostics).sort_values("combined_score", ascending=False).reset_index(drop=True)


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
aliases = prepare_ascii_aliases(SHP_DIR, TMP_ALIAS_DIR)

mask = load_layer(aliases["svita_new"])
facies = to_crs_safe(load_layer(aliases["fasii"]), mask.crs)
tect1 = to_crs_safe(load_layer(aliases["glub_raz_nw"]), mask.crs)
tect2 = to_crs_safe(load_layer(aliases["glub_r_nw"]), mask.crs)
paleo = to_crs_safe(load_layer(aliases["gr_dol_vp_poly"]), mask.crs)
struct = to_crs_safe(load_layer(aliases["kory"]), mask.crs)
magm = to_crs_safe(load_layer(aliases["dayki_buf"]), mask.crs)
points, point_info = collect_points(mask.crs, aliases)

print("Слои:", sorted(aliases))
print("Точечные слои-кандидаты:")
print(pd.DataFrame(point_info))


# =========================
# СЕТКА И ПРИЗНАКИ
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

grid["prox_facies"] = distance_to_proximity(grid["dist_facies"], "cbrt", 0.72)
grid["prox_paleo"] = distance_to_proximity(grid["dist_paleo"], "cbrt", 0.72)
grid["prox_struct"] = distance_to_proximity(grid["dist_struct"], "sqrt", 0.74)
grid["prox_magm"] = distance_to_proximity(grid["dist_magm"], "sqrt", 0.72)
grid["prox_tect1"] = distance_to_proximity(grid["dist_tect1"], "cbrt", 0.76)
grid["prox_tect2"] = distance_to_proximity(grid["dist_tect2"], "cbrt", 0.76)

grid["tect_combo"] = robust_normalize_01((grid["prox_tect1"] + grid["prox_tect2"]) / 2, 0.02, 0.98)
grid["tect_intersection"] = robust_normalize_01(grid["prox_tect1"] * grid["prox_tect2"], 0.02, 0.98)
grid["tect_magm_intersection"] = robust_normalize_01(grid["tect_combo"] * grid["prox_magm"], 0.02, 0.98)
grid["tect_struct_intersection"] = robust_normalize_01(grid["tect_combo"] * grid["prox_struct"], 0.02, 0.98)
grid["paleo_struct_intersection"] = robust_normalize_01(grid["prox_paleo"] * grid["prox_struct"], 0.02, 0.98)

grid["coincidence_score"] = robust_normalize_01(
    0.26 * grid["tect_combo"] +
    0.22 * grid["prox_struct"] +
    0.18 * grid["prox_paleo"] +
    0.18 * grid["prox_magm"] +
    0.16 * grid["prox_facies"],
    0.02, 0.98
)

grid["tect_only_penalty"] = robust_normalize_01(
    grid["tect_combo"] * (1 - 0.5 * grid["prox_struct"]) * (1 - 0.5 * grid["prox_paleo"]),
    0.02, 0.98
)

positive_feats = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_intersection", "tect_magm_intersection",
    "tect_struct_intersection", "paleo_struct_intersection", "coincidence_score"
]

grid["signal_base"] = robust_normalize_01(
    grid[positive_feats].mean(axis=1) - 0.35 * grid["tect_only_penalty"],
    0.02, 0.98
)

grid["signal_local"] = robust_normalize_01(
    smooth_on_regular_grid(grid, grid["signal_base"].to_numpy(), grid_shape, passes=1),
    0.02, 0.98
)

grid["tect_local"] = robust_normalize_01(
    smooth_on_regular_grid(grid, grid["tect_combo"].to_numpy(), grid_shape, passes=1),
    0.02, 0.98
)

grid["magm_struct_local"] = robust_normalize_01(
    smooth_on_regular_grid(grid, ((grid["prox_magm"] + grid["prox_struct"]) / 2).to_numpy(), grid_shape, passes=1),
    0.02, 0.98
)

grid["local_strength"] = robust_normalize_01(
    0.55 * grid["signal_base"] +
    0.25 * grid["signal_local"] +
    0.10 * grid["tect_local"] +
    0.10 * grid["magm_struct_local"],
    0.02, 0.98
)

print("Ячеек в сетке:", len(grid))
print("Размер сетки (rows, cols):", grid_shape)

feature_preview = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "coincidence_score",
    "tect_only_penalty", "signal_base", "signal_local", "local_strength"
]
print(grid[feature_preview].describe().T)


# =========================
# ПРОСТРАНСТВЕННАЯ КЛАСТЕРИЗАЦИЯ
# =========================
cluster_features = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_intersection", "tect_magm_intersection",
    "tect_struct_intersection", "paleo_struct_intersection", "coincidence_score",
    "tect_only_penalty", "signal_base", "signal_local", "tect_local",
    "magm_struct_local", "local_strength"
]

X = grid[cluster_features].to_numpy(dtype=float)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(PCA_COMPONENTS, X_scaled.shape[1]), random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

mask_matrix = np.zeros(grid_shape, dtype=bool)
mask_matrix[grid["row"].to_numpy(), grid["col"].to_numpy()] = True
connectivity = grid_to_graph(n_x=grid_shape[0], n_y=grid_shape[1], mask=mask_matrix)

if AUTO_SELECT_CLUSTERS:
    selected_k, labels, cluster_choice_df = choose_cluster_count(
        X_pca, connectivity, CLUSTER_CANDIDATES, random_state=RANDOM_STATE
    )
else:
    selected_k = int(N_CLUSTERS_FIXED)
    cluster_model = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage="ward",
        connectivity=connectivity,
    )
    labels = cluster_model.fit_predict(X_pca)
    cluster_choice_df = pd.DataFrame([
        {"k": selected_k, "silhouette": np.nan, "balance": np.nan, "entropy": np.nan, "combined_score": np.nan}
    ])

grid["cluster"] = labels.astype(int)

print("PCA explained variance:", round(float(pca.explained_variance_ratio_.sum()), 4))
print("Количество кластеров:", int(selected_k))
print("Размеры кластеров:", np.bincount(labels))
print(cluster_choice_df)


# =========================
# ПЕРЕВОД КЛАСТЕРОВ В НЕПРЕРЫВНУЮ ПЕРСПЕКТИВНОСТЬ
# =========================
cluster_summary = (
    grid.groupby("cluster")[cluster_features + ["signal_base", "signal_local", "local_strength", "tect_combo"]]
    .mean()
    .assign(cluster_size=grid.groupby("cluster").size())
    .reset_index()
)

cluster_summary["interaction_mean"] = cluster_summary[[
    "tect_intersection", "tect_magm_intersection", "tect_struct_intersection", "paleo_struct_intersection"
]].mean(axis=1)

cluster_summary["cluster_score_raw"] = (
    0.20 * cluster_summary["tect_combo"] +
    0.13 * cluster_summary["prox_struct"] +
    0.12 * cluster_summary["prox_paleo"] +
    0.12 * cluster_summary["prox_magm"] +
    0.08 * cluster_summary["prox_facies"] +
    0.15 * cluster_summary["interaction_mean"] +
    0.10 * cluster_summary["signal_base"] +
    0.10 * cluster_summary["signal_local"] -
    0.08 * cluster_summary["tect_only_penalty"]
)

cluster_summary["size_bonus"] = normalize_01(np.log1p(cluster_summary["cluster_size"]))
cluster_summary["cluster_score"] = robust_normalize_01(
    0.95 * cluster_summary["cluster_score_raw"] + 0.05 * cluster_summary["size_bonus"],
    0.02, 0.98
)

cluster_summary = cluster_summary.sort_values("cluster_score", ascending=False).reset_index(drop=True)
cluster_rank_map = cluster_summary.set_index("cluster")["cluster_score"].to_dict()

grid["cluster_base"] = grid["cluster"].map(cluster_rank_map).astype(float)
grid["cluster_percentile"] = grid.groupby("cluster")["local_strength"].rank(method="average", pct=True)

grid["prospectivity_raw"] = (
    0.50 * grid["cluster_base"] +
    0.20 * grid["cluster_percentile"] +
    0.30 * grid["local_strength"]
)

grid["prospectivity_sm"] = smooth_on_regular_grid(
    grid,
    grid["prospectivity_raw"].to_numpy(),
    grid_shape,
    passes=SMOOTH_PASSES,
)

grid["prospectivity"] = robust_normalize_01(grid["prospectivity_sm"], 0.02, 0.98)
grid["prognoz"] = 1.0 - grid["prospectivity"]

grid = mark_top_zones(grid, grid_shape)
grid = make_display_classes(grid)

print(cluster_summary[["cluster", "cluster_size", "cluster_score", "cluster_score_raw"]])


# =========================
# ПРОВЕРКА ПО ИЗВЕСТНЫМ ТОЧКАМ
# =========================
positive_cells = []
if points is not None and len(points) > 0:
    try:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", op="within")
    positive_cells = joined["cell_id"].dropna().astype(int).unique().tolist()

try:
    sample_size = min(5000, len(X_pca))
    rs = np.random.RandomState(RANDOM_STATE)
    sample_idx = rs.choice(len(X_pca), size=sample_size, replace=False)
    silhouette = float(silhouette_score(X_pca[sample_idx], labels[sample_idx])) if len(np.unique(labels[sample_idx])) > 1 else np.nan
except Exception:
    silhouette = np.nan

metrics = {
    "n_cells": int(len(grid)),
    "n_clusters": int(selected_k),
    "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    "silhouette_sample": silhouette,
    "positive_cells_from_points": int(len(positive_cells)),
    "smooth_passes": int(SMOOTH_PASSES),
    "auto_select_clusters": bool(AUTO_SELECT_CLUSTERS),
}

rows = []
for q in [0.90, 0.85, 0.80]:
    thr = float(grid["prospectivity"].quantile(q))
    top_mask = grid["prospectivity"] >= thr
    area_share = float(top_mask.mean())

    if positive_cells:
        hit_mask = grid.loc[grid["cell_id"].isin(positive_cells), "prospectivity"] >= thr
        hit_rate = float(hit_mask.mean()) if len(hit_mask) else np.nan
        lift = float(hit_rate / area_share) if area_share > 0 else np.nan
    else:
        hit_rate = np.nan
        lift = np.nan

    top_pct = int(round((1 - q) * 100))
    metrics[f"hit_rate_top_{top_pct:02d}pct"] = hit_rate
    metrics[f"lift_top_{top_pct:02d}pct"] = lift

    rows.append({
        "Top zone": f"Top {top_pct}%",
        "Threshold": thr,
        "Area share": area_share,
        "Hit rate": hit_rate,
        "Lift": lift,
    })

# Отдельная оценка именно жёлтых узлов
if positive_cells:
    top_zone_cell_ids = set(grid.loc[grid["top_zone"] == 1, "cell_id"].astype(int).tolist())
    overlap = len(top_zone_cell_ids.intersection(set(positive_cells)))
    metrics["top_zone_hits"] = int(overlap)
    metrics["top_zone_cells"] = int(len(top_zone_cell_ids))
    metrics["top_zone_hit_rate"] = float(overlap / len(positive_cells)) if len(positive_cells) else np.nan

metrics_df = pd.DataFrame(rows)
print(metrics_df)


# =========================
# СОХРАНЕНИЕ
# =========================
OUT_GPKG = OUT_DIR / "spatial_agglomerative_forecast_v2.gpkg"
OUT_PNG = OUT_DIR / "spatial_agglomerative_forecast_v2.png"
OUT_CLUSTER_CSV = OUT_DIR / "cluster_summary_v2.csv"
OUT_CLUSTER_DIAGNOSTICS = OUT_DIR / "cluster_choice_diagnostics_v2.csv"
OUT_GRID_CSV = OUT_DIR / "forecast_grid_attributes_v2.csv"
OUT_METRICS = OUT_DIR / "metrics_v2.json"
OUT_PARAMS = OUT_DIR / "run_params_v2.json"

plot_final(grid, mask, points, OUT_PNG)

grid.to_file(OUT_GPKG, layer="forecast_grid", driver="GPKG")
cluster_summary.to_csv(OUT_CLUSTER_CSV, index=False)
cluster_choice_df.to_csv(OUT_CLUSTER_DIAGNOSTICS, index=False)
grid.drop(columns="geometry").to_csv(OUT_GRID_CSV, index=False)

with open(OUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open(OUT_PARAMS, "w", encoding="utf-8") as f:
    json.dump({
        "CELL_SIZE": CELL_SIZE,
        "AUTO_SELECT_CLUSTERS": AUTO_SELECT_CLUSTERS,
        "CLUSTER_CANDIDATES": CLUSTER_CANDIDATES,
        "N_CLUSTERS_FIXED": N_CLUSTERS_FIXED,
        "PCA_COMPONENTS": PCA_COMPONENTS,
        "SMOOTH_PASSES": SMOOTH_PASSES,
        "TOP_ZONE_CORE_Q": TOP_ZONE_CORE_Q,
        "TOP_ZONE_SIGNAL_Q": TOP_ZONE_SIGNAL_Q,
        "TOP_ZONE_CLUSTER_PCT_Q": TOP_ZONE_CLUSTER_PCT_Q,
        "LOCAL_PEAK_SIZE": LOCAL_PEAK_SIZE,
        "MIN_TOP_ZONE_FRACTION": MIN_TOP_ZONE_FRACTION,
    }, f, ensure_ascii=False, indent=2)

print("PNG:", OUT_PNG)
print("GPKG:", OUT_GPKG)
print("CLUSTER CSV:", OUT_CLUSTER_CSV)
print("GRID CSV:", OUT_GRID_CSV)
print("DIAGNOSTICS CSV:", OUT_CLUSTER_DIAGNOSTICS)
print("METRICS JSON:", OUT_METRICS)
print("PARAMS JSON:", OUT_PARAMS)
