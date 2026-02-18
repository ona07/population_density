# Japan mesh choropleth script (renamed from donan.py)

import io
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

try:
    from shapely import make_valid as _make_valid
except Exception:  # pragma: no cover
    _make_valid = None

def meshcode3_bounds(code: str) -> tuple[float, float, float, float]:
    """
    JIS X 0410 3次メッシュ（8桁, 約1km）コードの範囲（経度/緯度）を返す。

    Returns:
        (west, south, east, north) in degrees (lon/lat).
    """
    s = str(code).strip()
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Expected 8-digit 3rd-level mesh code, got: {code!r}")

    lat1 = int(s[0:2]) * (40.0 / 60.0)
    lon1 = int(s[2:4]) + 100.0

    lat2 = int(s[4]) * (5.0 / 60.0)
    lon2 = int(s[5]) * (7.5 / 60.0)

    lat3 = int(s[6]) * (30.0 / 3600.0)
    lon3 = int(s[7]) * (45.0 / 3600.0)

    south = lat1 + lat2 + lat3
    west = lon1 + lon2 + lon3
    north = south + (30.0 / 3600.0)
    east = west + (45.0 / 3600.0)
    return west, south, east, north

def meshcode2_bounds(code: str) -> tuple[float, float, float, float]:
    """
    JIS X 0410 2次メッシュ（6桁）コードの範囲（経度/緯度）を返す。

    Returns:
        (west, south, east, north) in degrees (lon/lat).
    """
    s = str(code).strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"Expected 6-digit 2nd-level mesh code, got: {code!r}")

    lat1 = int(s[0:2]) * (40.0 / 60.0)
    lon1 = int(s[2:4]) + 100.0
    lat2 = int(s[4]) * (5.0 / 60.0)
    lon2 = int(s[5]) * (7.5 / 60.0)

    south = lat1 + lat2
    west = lon1 + lon2
    north = south + (5.0 / 60.0)
    east = west + (7.5 / 60.0)
    return west, south, east, north

def meshcode2_sub_bounds(code: str, *, subdiv: int) -> tuple[float, float, float, float]:
    """
    2次メッシュ（6桁）を subdiv×subdiv に分割した擬似コード（例: 533945-01）の範囲を返す。

    subdiv は 10 を割り切る値のみ対応（例: 2=約5km, 5=約2km）。

    Returns:
        (west, south, east, north) in degrees (lon/lat).
    """
    s = str(code).strip()
    if len(s) != 9 or s[6] != "-" or (not s[:6].isdigit()) or (not s[7:9].isdigit()):
        raise ValueError(f"Expected mesh2-subdiv code like 'XXXXXX-ij', got: {code!r}")

    subdiv = int(subdiv)
    if subdiv <= 1 or 10 % subdiv != 0:
        raise ValueError("subdiv must be a divisor of 10 (e.g. 2,5)")

    mesh2 = s[:6]
    i = int(s[7])
    j = int(s[8])
    if not (0 <= i < subdiv) or not (0 <= j < subdiv):
        raise ValueError(f"Expected i,j in 0..{subdiv-1} for mesh2 subdiv, got: {code!r}")

    west0, south0, east0, north0 = meshcode2_bounds(mesh2)
    lat_step = (north0 - south0) / subdiv
    lon_step = (east0 - west0) / subdiv
    south = south0 + i * lat_step
    north = south + lat_step
    west = west0 + j * lon_step
    east = west + lon_step
    return west, south, east, north

def meshcode1_bounds(code: str) -> tuple[float, float, float, float]:
    """
    JIS X 0410 1次メッシュ（4桁）コードの範囲（経度/緯度）を返す。

    Returns:
        (west, south, east, north) in degrees (lon/lat).
    """
    s = str(code).strip()
    if len(s) != 4 or not s.isdigit():
        raise ValueError(f"Expected 4-digit 1st-level mesh code, got: {code!r}")

    south = int(s[0:2]) * (40.0 / 60.0)
    west = int(s[2:4]) + 100.0
    north = south + (40.0 / 60.0)
    east = west + 1.0
    return west, south, east, north

def _make_geometry_valid(geom):
    if geom is None:
        return None
    if _make_valid is not None:
        try:
            return _make_valid(geom)
        except Exception:
            pass
    try:
        return geom.buffer(0)
    except Exception:
        return geom

def load_land_union_naturalearth(
    *,
    bbox_wgs84: tuple[float, float, float, float],
    scale: str = "50m",
    cache_dir: Path = Path(".cache") / "naturalearth",
) -> object | None:
    """
    Natural Earth の land ポリゴンをダウンロードして、対象範囲の陸域 union を返す。
    失敗した場合は None を返す（海の切り抜きを諦めるフォールバック用）。
    """
    scale = str(scale).strip()
    if scale not in {"10m", "50m", "110m"}:
        raise ValueError("scale must be one of: 10m, 50m, 110m")

    url = f"https://naturalearth.s3.amazonaws.com/{scale}_physical/ne_{scale}_land.zip"
    dest_dir = cache_dir / f"ne_{scale}_land"
    shp_path = dest_dir / f"ne_{scale}_land.shp"
    zip_path = dest_dir / f"ne_{scale}_land.zip"

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not shp_path.exists():
            if not zip_path.exists():
                print(f"download: Natural Earth land ({scale})")
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest_dir)

        gdf = gpd.read_file(shp_path, bbox=bbox_wgs84)
        if gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        land = gdf.geometry.union_all()
        land = _make_geometry_valid(land)
        return land
    except Exception as e:
        print(f"warn: failed to load Natural Earth land polygons ({scale}): {e}")
        return None

def load_country_union_naturalearth(
    *,
    bbox_wgs84: tuple[float, float, float, float],
    country_iso_a3: str,
    scale: str = "10m",
    cache_dir: Path = Path(".cache") / "naturalearth",
) -> object | None:
    """
    Natural Earth の admin_0_countries から国境ポリゴンを読み込み、指定国の union を返す。
    失敗した場合は None を返す。
    """
    scale = str(scale).strip()
    if scale not in {"10m", "50m", "110m"}:
        raise ValueError("scale must be one of: 10m, 50m, 110m")

    iso = str(country_iso_a3).strip().upper()
    if len(iso) != 3:
        raise ValueError("country_iso_a3 must be ISO A3 code like 'JPN'")

    url = (
        f"https://naturalearth.s3.amazonaws.com/{scale}_cultural/"
        f"ne_{scale}_admin_0_countries.zip"
    )
    dest_dir = cache_dir / f"ne_{scale}_admin_0_countries"
    shp_path = dest_dir / f"ne_{scale}_admin_0_countries.shp"
    zip_path = dest_dir / f"ne_{scale}_admin_0_countries.zip"

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not shp_path.exists():
            if not zip_path.exists():
                print(f"download: Natural Earth admin_0_countries ({scale})")
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest_dir)

        gdf = gpd.read_file(shp_path, bbox=bbox_wgs84)
        if gdf.empty:
            # bbox が狭すぎた等で取りこぼすことがあるので全読みしてみる
            gdf = gpd.read_file(shp_path)
        if gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        iso_col = None
        for c in ("ISO_A3", "ADM0_A3", "SOV_A3", "GU_A3", "SU_A3", "BRK_A3"):
            if c in gdf.columns:
                iso_col = c
                break

        if iso_col is not None:
            mask = gdf[iso_col].astype(str).str.upper().eq(iso)
        else:
            name_cols = [c for c in ("ADMIN", "NAME", "SOVEREIGNT", "NAME_LONG") if c in gdf.columns]
            mask = False
            for c in name_cols:
                mask = mask | gdf[c].astype(str).str.strip().str.casefold().eq("japan")

        country = gdf.loc[mask]
        if country.empty:
            return None
        geom = country.geometry.union_all()
        geom = _make_geometry_valid(geom)
        return geom
    except Exception as e:
        print(f"warn: failed to load Natural Earth country polygon ({iso}, {scale}): {e}")
        return None


def load_populated_places_naturalearth(
    *,
    bbox_wgs84: tuple[float, float, float, float],
    scale: str = "10m",
    cache_dir: Path = Path(".cache") / "naturalearth",
) -> gpd.GeoDataFrame | None:
    """
    Natural Earth の populated places をダウンロードして GeoDataFrame で返す。
    失敗した場合は None を返す。
    """
    scale = str(scale).strip()
    if scale not in {"10m", "50m", "110m"}:
        raise ValueError("scale must be one of: 10m, 50m, 110m")

    url = (
        f"https://naturalearth.s3.amazonaws.com/{scale}_cultural/"
        f"ne_{scale}_populated_places.zip"
    )
    dest_dir = cache_dir / f"ne_{scale}_populated_places"
    shp_path = dest_dir / f"ne_{scale}_populated_places.shp"
    zip_path = dest_dir / f"ne_{scale}_populated_places.zip"

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not shp_path.exists():
            if not zip_path.exists():
                print(f"download: Natural Earth populated_places ({scale})")
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest_dir)

        gdf = gpd.read_file(shp_path, bbox=bbox_wgs84)
        if gdf.empty:
            gdf = gpd.read_file(shp_path)
        if gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        print(f"warn: failed to load Natural Earth populated places ({scale}): {e}")
        return None


def select_city_labels(
    *,
    bbox_wgs84: tuple[float, float, float, float],
    region_wgs84: object | None,
    max_labels: int,
    min_sep_km: float,
    scale: str = "10m",
) -> list[dict]:
    """
    日本の主要都市ラベルを間引いて返す。

    Returns:
        [{"name": str, "lon": float, "lat": float, "x": float, "y": float}, ...]
        x,y は EPSG:3857（PNG描画用）。
    """
    if max_labels <= 0:
        return []

    cities = load_populated_places_naturalearth(bbox_wgs84=bbox_wgs84, scale=scale)
    if cities is None or cities.empty:
        return []

    cities_wgs84 = cities
    if region_wgs84 is not None:
        try:
            region_valid = _make_geometry_valid(region_wgs84)
            cities_wgs84 = cities_wgs84.loc[cities_wgs84.geometry.intersects(region_valid)].copy()
        except Exception:
            pass
    else:
        iso_cols = [c for c in ("ADM0_A3", "SOV_A3", "ISO_A3") if c in cities_wgs84.columns]
        if iso_cols:
            cities_wgs84 = cities_wgs84.loc[
                cities_wgs84[iso_cols[0]].astype(str).str.upper().eq("JPN")
            ].copy()

    if cities_wgs84.empty:
        return []

    name_col = None
    for c in ("NAME", "name", "NAMEASCII", "nameascii", "NAME_EN", "name_en"):
        if c in cities_wgs84.columns:
            name_col = c
            break
    if name_col is None:
        return []

    pop_col = None
    for c in ("POP_MAX", "pop_max", "POP_MIN", "pop_min"):
        if c in cities_wgs84.columns:
            pop_col = c
            break

    cities_wgs84["_name"] = cities_wgs84[name_col].astype(str).str.strip()
    cities_wgs84 = cities_wgs84.loc[cities_wgs84["_name"].ne("")].copy()
    if cities_wgs84.empty:
        return []

    cities_wgs84["_pop"] = pd.to_numeric(
        cities_wgs84[pop_col], errors="coerce"
    ) if pop_col else 0
    cities_wgs84["_pop"] = cities_wgs84["_pop"].fillna(0)

    # FEATURECLA で「首都/州都」を優先
    if "FEATURECLA" in cities_wgs84.columns:
        fc = cities_wgs84["FEATURECLA"].astype(str).str.casefold()
        cities_wgs84["_rank"] = 0.0
        cities_wgs84.loc[fc.str.contains("admin-0"), "_rank"] += 1000.0
        cities_wgs84.loc[fc.str.contains("admin-1"), "_rank"] += 500.0
    else:
        cities_wgs84["_rank"] = 0.0

    cities_wgs84 = cities_wgs84.sort_values(by=["_rank", "_pop"], ascending=[False, False])

    cities_3857 = cities_wgs84.to_crs("EPSG:3857")
    min_sep_m = float(min_sep_km) * 1000.0
    min_sep2 = min_sep_m * min_sep_m

    out: list[dict] = []
    keep_xy: list[tuple[float, float]] = []
    seen: set[str] = set()
    for idx, row_city in cities_3857.iterrows():
        if len(out) >= int(max_labels):
            break

        name = str(row_city.get("_name", "")).strip()
        if not name or name in seen:
            continue

        geom = row_city.geometry
        if geom is None or geom.is_empty:
            continue
        x = float(getattr(geom, "x", np.nan))
        y = float(getattr(geom, "y", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if keep_xy:
            ok = True
            for x0, y0 in keep_xy:
                dx0 = x - x0
                dy0 = y - y0
                if (dx0 * dx0 + dy0 * dy0) < min_sep2:
                    ok = False
                    break
            if not ok:
                continue

        geom0 = cities_wgs84.loc[idx].geometry
        if geom0 is None or geom0.is_empty:
            continue
        lon = float(getattr(geom0, "x", np.nan))
        lat = float(getattr(geom0, "y", np.nan))
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue

        out.append({"name": name, "lon": lon, "lat": lat, "x": x, "y": y})
        keep_xy.append((x, y))
        seen.add(name)

    return out

def iter_meshcodes2_in_first_mesh(first_code: str):
    """
    1次メッシュ（4桁）配下の 2次メッシュ（6桁）コードをすべて列挙する。

    1次: 40' × 1°
    2次: 5' × 7.5'（各8分割）
    """
    s = str(first_code).strip()
    if len(s) != 4 or not s.isdigit():
        raise ValueError(f"Expected 4-digit 1st-level mesh code, got: {first_code!r}")

    for a in range(8):
        for b in range(8):
            yield f"{s}{a}{b}"

def iter_meshcodes3_in_second_mesh(second_code: str):
    """
    2次メッシュ（6桁）配下の 3次メッシュ（8桁）コードをすべて列挙する。

    2次: 5' × 7.5'（各10分割）
    3次: 30" × 45"
    """
    s = str(second_code).strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"Expected 6-digit 2nd-level mesh code, got: {second_code!r}")

    for c in range(10):
        for d in range(10):
            yield f"{s}{c}{d}"


def _grid_xy_from_meshcode2(code: str, *, min_lat: int, min_lon: int) -> tuple[int, int]:
    # 2次メッシュ 6桁: YYXXab（a,b:2次）
    s = str(code).strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"Expected 6-digit 2nd-level mesh code, got: {code!r}")

    lat = int(s[0:2])
    lon = int(s[2:4])
    a = int(s[4])
    b = int(s[5])
    y = (lat - min_lat) * 8 + a
    x = (lon - min_lon) * 8 + b
    return y, x

def _fill_holes_mask_from_seed_mesh2(
    df: pd.DataFrame,
    *,
    code_col: str,
    seed_col: str,
    codes4: list[str],
    dilate_steps: int = 0,
) -> pd.Series:
    codes4 = [str(c).strip().zfill(4) for c in codes4]
    lats = [int(c[:2]) for c in codes4]
    lons = [int(c[2:4]) for c in codes4]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    height = (max_lat - min_lat + 1) * 8
    width = (max_lon - min_lon + 1) * 8
    seed = np.zeros((height, width), dtype=bool)

    ys = np.empty(len(df), dtype=np.int32)
    xs = np.empty(len(df), dtype=np.int32)
    codes = df[code_col].astype(str).tolist()
    for i, code in enumerate(codes):
        y, x = _grid_xy_from_meshcode2(code, min_lat=min_lat, min_lon=min_lon)
        ys[i] = y
        xs[i] = x

    seed_rows = df[seed_col].fillna(False).to_numpy(dtype=bool)
    seed[ys[seed_rows], xs[seed_rows]] = True

    if dilate_steps > 0 and seed.any():
        dist = np.full(seed.shape, fill_value=-1, dtype=np.int16)
        q: deque[tuple[int, int]] = deque()
        sy, sx = np.where(seed)
        for y0, x0 in zip(sy.tolist(), sx.tolist()):
            dist[y0, x0] = 0
            q.append((y0, x0))

        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
        while q:
            y, x = q.popleft()
            d = dist[y, x]
            if d >= dilate_steps:
                continue
            nd = d + 1
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if dist[ny, nx] != -1:
                    continue
                dist[ny, nx] = nd
                q.append((ny, nx))
        seed = dist != -1

    outside = np.zeros_like(seed, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    # 境界から「種(seed)ではないセル」を flood fill して外側(=海側候補)を作る
    for x in range(width):
        if not seed[0, x]:
            outside[0, x] = True
            q.append((0, x))
        if not seed[height - 1, x]:
            outside[height - 1, x] = True
            q.append((height - 1, x))
    for y in range(height):
        if not seed[y, 0]:
            outside[y, 0] = True
            q.append((y, 0))
        if not seed[y, width - 1]:
            outside[y, width - 1] = True
            q.append((y, width - 1))

    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    while q:
        y, x = q.popleft()
        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if seed[ny, nx] or outside[ny, nx]:
                continue
            outside[ny, nx] = True
            q.append((ny, nx))

    include_rows = ~outside[ys, xs]
    return pd.Series(include_rows, index=df.index)

BASE = "https://www.e-stat.go.jp/gis/statmap-search/data"
STATSID = "T001140"  # 2020 国勢調査 3次メッシュ人口（JGD2011）

# donan(道南)用に残しておく（必要なら REGION を "donan" に）
REGION = "japan"  # "donan" | "japan"
OUTPUT_HTML = "donan_choropleth.html" if REGION == "donan" else "japan_choropleth.html"
# 全国はポリゴンHTMLだと巨大になりがちなので、既定はラスタ重ね（軽量なインタラクティブHTML）
OUTPUT_MODE = "html" if REGION == "donan" else "raster_html"  # "html" | "png" | "both" | "raster_html"
OUTPUT_PNG = "donan_density.png" if REGION == "donan" else "japan_density.png"
OUTPUT_RASTER_PNG = "donan_overlay.png" if REGION == "donan" else "japan_overlay.png"
OUTPUT_RASTER_HTML = "donan_interactive.html" if REGION == "donan" else "japan_interactive.html"
# 日本全国は既定でラスタ出力なので、デフォルトは 3次メッシュ(約1km)まで上げる
# ※HTML化する場合は重くなりやすいので OUTPUT_MESH_LEVEL=2 も検討
OUTPUT_MESH_LEVEL = 3 if REGION == "donan" else 3  # 2 or 3
# 2次メッシュ出力時の分割数（10を割り切る値のみ: 1=約10km, 2=約5km, 5=約2km）
OUTPUT_MESH2_SUBDIV = 1 if REGION == "donan" else 5
PNG_LONG_SIDE_PX = 8000
PNG_DPI = 400
FILL_ALPHA = 0.55

# PNG上に主要都市ラベルを描画（Natural Earth populated places を利用）
PLOT_CITY_LABELS = True
CITY_LABEL_MAX = 45 if REGION == "japan" else 20
CITY_MIN_SEP_KM = 25.0  # ラベルの間隔（だいたいの最小距離）
CITY_DOT_SIZE = 6
CITY_LABEL_FONT_SIZE = None  # None=自動, 数値=固定
CITY_LABEL_FONT_PX = 12  # raster_html のラベル表示用
codes_donan = [
    "6139",
    "6140",
    "6141",
    "6142",
    "6239",
    "6240",
    "6241",
    "6242",
    "6339",
    "6340",
    "6341",
    "6342",
    "6439",
    "6440",
    "6441",
    "6442",
]

def mesh1_codes_for_japan() -> list[str]:
    """
    日本（概ね）の範囲を覆う 1次メッシュ（4桁）コードを列挙する（朝鮮半島などは除外）。
    ※e-Stat 側で 404 はスキップする。
    """
    min_lon, min_lat, max_lon, max_lat = 122.0, 20.0, 154.0, 46.0

    lat_start = int(np.floor(min_lat * 1.5))
    lat_end = int(np.floor(max_lat * 1.5))
    lon_start = int(np.floor(min_lon)) - 100
    lon_end = int(np.floor(max_lon)) - 100

    out: list[str] = []
    for lat_code in range(lat_start, lat_end + 1):
        for lon_code in range(lon_start, lon_end + 1):
            out.append(f"{lat_code:02d}{lon_code:02d}")

    bbox = (min_lon, min_lat, max_lon, max_lat)
    japan = load_country_union_naturalearth(
        bbox_wgs84=bbox,
        country_iso_a3="JPN",
        scale="10m",
    )
    if japan is None:
        return out

    japan_prep = prep(japan)
    keep: list[str] = []
    for code4 in out:
        w, s, e, n = meshcode1_bounds(code4)
        if japan_prep.intersects(box(w, s, e, n)):
            keep.append(code4)
    return keep

codes_mesh1 = codes_donan if REGION == "donan" else mesh1_codes_for_japan()
ESTAT_CACHE_DIR = Path(".cache") / "estat" / STATSID
ESTAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _read_table_from_zip(z: zipfile.ZipFile, *, _depth: int = 0) -> pd.DataFrame:
    if _depth > 2:
        raise RuntimeError("Too many nested zip levels while searching for data files.")

    names = [n for n in z.namelist() if not n.endswith("/")]

    def read_csv_bytes(data: bytes, **kwargs) -> pd.DataFrame:
        for enc in ("cp932", "utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.BytesIO(data), encoding="cp932", **kwargs)

    csv_names = [n for n in names if n.lower().endswith(".csv")]
    if csv_names:
        return read_csv_bytes(z.read(csv_names[0]))

    tsv_names = [n for n in names if n.lower().endswith((".tsv", ".txt"))]
    if tsv_names:
        return read_csv_bytes(z.read(tsv_names[0]), sep=None, engine="python")

    shp_names = [n for n in names if n.lower().endswith(".shp")]
    if shp_names:
        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            shp_path = Path(td) / shp_names[0]
            gdf = gpd.read_file(shp_path)
            if "geometry" in gdf.columns:
                gdf = gdf.drop(columns=["geometry"])
            return pd.DataFrame(gdf)

    nested_zip_names = [n for n in names if n.lower().endswith(".zip")]
    for inner_name in nested_zip_names:
        inner_bytes = z.read(inner_name)
        if not zipfile.is_zipfile(io.BytesIO(inner_bytes)):
            continue
        inner_zip = zipfile.ZipFile(io.BytesIO(inner_bytes))
        try:
            return _read_table_from_zip(inner_zip, _depth=_depth + 1)
        except Exception:
            continue

    raise RuntimeError(
        "No supported data file found in downloaded zip. "
        f"Files: {names[:30]}{' ...' if len(names) > 30 else ''}"
    )


def download_mesh_table(code: str, *, session: requests.Session) -> pd.DataFrame | None:
    params = {"statsId": STATSID, "code": code, "downloadType": 2}
    cache_path = ESTAT_CACHE_DIR / f"{code}.bin"
    notfound_path = ESTAT_CACHE_DIR / f"{code}.404"

    if notfound_path.exists():
        return None

    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        for attempt in range(5):
            r = session.get(BASE, params=params, timeout=60)
            if r.status_code == 404:
                notfound_path.touch()
                return None
            if r.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(int(retry_after))
            else:
                time.sleep(2**attempt)
        r.raise_for_status()

        content = r.content
        cache_path.write_bytes(content)

    # e-Stat は zip で来ることが多い
    if zipfile.is_zipfile(io.BytesIO(content)):
        z = zipfile.ZipFile(io.BytesIO(content))
        df = _read_table_from_zip(z)
    else:
        # まれに zip ではなく CSV が直接返るケースに備える
        for enc in ("cp932", "utf-8-sig", "utf-8"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(io.BytesIO(content), encoding="cp932")

    df["source_code"] = code  # どの第1次地域区画から来たか
    return df

def _series_as_clean_str(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def infer_meshcode_column(df: pd.DataFrame) -> object:
    candidates: list[tuple[int, object]] = []
    for c in df.columns:
        name = str(c)
        vals = _series_as_clean_str(df[c]).replace({"nan": pd.NA})
        vals = vals.dropna()
        if vals.empty:
            continue
        is_8digit = vals.str.match(r"^\d{8}$", na=False)
        if is_8digit.mean() < 0.8:
            continue

        score = 0
        low = name.lower()
        if "mesh" in low:
            score += 3
        if "code" in low:
            score += 1
        if "コード" in name:
            score += 3
        candidates.append((score, c))

    if not candidates:
        raise RuntimeError(
            "Could not infer mesh code column (expected mostly 8-digit codes). "
            f"Columns: {list(df.columns)}"
        )

    best = max(candidates, key=lambda x: (x[0], str(x[1])))
    return best[1]


def infer_population_column(df: pd.DataFrame, *, mesh_col: str) -> object:
    numeric_cols: list[object] = []
    for c in df.columns:
        if c in {mesh_col, "source_code"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            continue

        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.8:
            df[c] = coerced
            numeric_cols.append(c)

    if not numeric_cols:
        raise RuntimeError(
            "Could not find numeric population column. "
            f"Columns: {list(df.columns)}"
        )

    candidates: list[tuple[int, object]] = []
    for c in numeric_cols:
        name = str(c)
        low = name.lower()
        score = 0
        if STATSID in name:
            score += 5
        if "総人口" in name:
            score += 5
        if "人口" in name:
            score += 3
        if "pop" in low:
            score += 3
        if "male" in low or "female" in low or "男" in name or "女" in name:
            score -= 2
        candidates.append((score, c))

    best = max(candidates, key=lambda x: (x[0], str(x[1])))
    return best[1]


dfs: list[pd.DataFrame] = []
with requests.Session() as session:
    for code in codes_mesh1:
        part = download_mesh_table(code, session=session)
        if part is None:
            print(f"skip: code={code} はデータが見つかりませんでした (HTTP 404)")
            continue
        if not part.empty:
            dfs.append(part)

if not dfs:
    raise RuntimeError("No data downloaded. STATSID/codes_mesh1 を見直してください。")

df = pd.concat(dfs, ignore_index=True)

# メッシュコード列を推定（名前が調査によって変わることがある）
mesh_col = infer_meshcode_column(df)
df = df.rename(columns={mesh_col: "meshcode"}).copy()
df["meshcode"] = _series_as_clean_str(df["meshcode"])

# 人口列を推定（たいてい  '総人口' 的な列がある）
pop_col = infer_population_column(df, mesh_col="meshcode")
df = df.rename(columns={pop_col: "population"}).copy()
df["population"] = pd.to_numeric(df["population"], errors="coerce")

mesh_ok = df["meshcode"].str.match(r"^\d{8}$", na=False)
if not mesh_ok.all():
    bad_examples = df.loc[~mesh_ok, "meshcode"].head(5).tolist()
    print(
        "Skipping rows with invalid meshcode:",
        int((~mesh_ok).sum()),
        "e.g.",
        bad_examples,
    )
    df = df.loc[mesh_ok].copy()

# e-Stat 側に行が存在しないメッシュも黒で塗れるよう、陸上メッシュを補完する
FILL_MISSING_MESHES = True
if FILL_MISSING_MESHES:
    base_mesh3 = df[["meshcode", "population"]].copy()
    base_mesh3 = base_mesh3.drop_duplicates(subset=["meshcode"]).copy()
    base_mesh3["mesh2"] = base_mesh3["meshcode"].astype(str).str.slice(0, 6)

    mesh2_all: list[str] = []
    for code4 in codes_mesh1:
        mesh2_all.extend(iter_meshcodes2_in_first_mesh(code4))

    # 海・国外を塗らないために「日本ポリゴン」で切り抜く（Natural Earth admin0 を利用）
    USE_JAPAN_POLYGON = True
    region_wgs84 = None
    if USE_JAPAN_POLYGON:
        xs, ys = [], []
        for code4 in codes_mesh1:
            w, s, e, n = meshcode1_bounds(code4)
            xs.extend([w, e])
            ys.extend([s, n])
        bbox = (min(xs), min(ys), max(xs), max(ys))
        region_wgs84 = load_country_union_naturalearth(
            bbox_wgs84=bbox,
            country_iso_a3="JPN",
            scale="10m",
        )

    if region_wgs84 is not None:
        region_prep = prep(region_wgs84)
        mesh2_keep: list[str] = []
        for code6 in mesh2_all:
            w, s, e, n = meshcode2_bounds(code6)
            if region_prep.intersects(box(w, s, e, n)):
                mesh2_keep.append(code6)
        print("mesh2 kept (japan mask):", len(mesh2_keep), "/", len(mesh2_all))
    else:
        mesh2_df = pd.DataFrame({"mesh2": mesh2_all})
        mesh2_df["has_data2"] = mesh2_df["mesh2"].isin(set(base_mesh3["mesh2"].unique()))

        # 2次メッシュのデータ存在セルを「陸の種」とみなし、外周から到達可能な領域（海側）を落とす
        SEA_CUT_DILATE2 = 1
        keep2 = _fill_holes_mask_from_seed_mesh2(
            mesh2_df,
            code_col="mesh2",
            seed_col="has_data2",
            codes4=codes_mesh1,
            dilate_steps=SEA_CUT_DILATE2,
        )
        mesh2_keep = mesh2_df.loc[keep2, "mesh2"].tolist()
        print(
            "mesh2 kept (seed-fill fallback):",
            len(mesh2_keep),
            "/",
            len(mesh2_df),
            "(has data:",
            int(mesh2_df["has_data2"].sum()),
            ")",
        )

    if OUTPUT_MESH_LEVEL == 3:
        mesh3_all: list[str] = []
        for code6 in mesh2_keep:
            mesh3_all.extend(iter_meshcodes3_in_second_mesh(code6))

        full = pd.DataFrame({"meshcode": mesh3_all})
        df = full.merge(base_mesh3[["meshcode", "population"]], on="meshcode", how="left")
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        print(
            "mesh3 total:",
            len(df),
            "missing population:",
            int(df["population"].isna().sum()),
            "pop<=0:",
            int((df["population"] <= 0).sum()),
        )
    elif OUTPUT_MESH_LEVEL == 2:
        if OUTPUT_MESH2_SUBDIV == 1:
            pop2 = (
                base_mesh3.groupby("mesh2", as_index=False)["population"]
                .sum(min_count=1)
                .rename(columns={"mesh2": "meshcode"})
            )
            full = pd.DataFrame({"meshcode": mesh2_keep})
            df = full.merge(pop2, on="meshcode", how="left")
            df["population"] = pd.to_numeric(df["population"], errors="coerce")
            print(
                "mesh2 total:",
                len(df),
                "missing population:",
                int(df["population"].isna().sum()),
                "pop<=0:",
                int((df["population"] <= 0).sum()),
            )
        else:
            subdiv = int(OUTPUT_MESH2_SUBDIV)
            if subdiv <= 1 or 10 % subdiv != 0:
                raise ValueError("OUTPUT_MESH2_SUBDIV must be a divisor of 10 (e.g. 1,2,5)")

            # 2次メッシュを subdiv×subdiv に分割（約10km/subdiv）して 3次メッシュ人口を合算する
            bin_size = 10 // subdiv

            base_sub = base_mesh3.copy()
            c = base_sub["meshcode"].astype(str).str.slice(6, 7).astype(int)
            d = base_sub["meshcode"].astype(str).str.slice(7, 8).astype(int)
            base_sub["meshcode_sub"] = (
                base_sub["mesh2"].astype(str)
                + "-"
                + (c // bin_size).astype(str)
                + (d // bin_size).astype(str)
            )
            pop_sub = (
                base_sub.groupby("meshcode_sub", as_index=False)["population"]
                .sum(min_count=1)
                .rename(columns={"meshcode_sub": "meshcode"})
            )

            sub_all: list[str] = []
            for code6 in mesh2_keep:
                for i in range(subdiv):
                    for j in range(subdiv):
                        sub_all.append(f"{code6}-{i}{j}")

            full = pd.DataFrame({"meshcode": sub_all})
            df = full.merge(pop_sub, on="meshcode", how="left")
            df["population"] = pd.to_numeric(df["population"], errors="coerce")
            print(
                f"mesh2/{subdiv} total:",
                len(df),
                "missing population:",
                int(df["population"].isna().sum()),
                "pop<=0:",
                int((df["population"] <= 0).sum()),
            )
    else:
        raise ValueError("OUTPUT_MESH_LEVEL must be 2 or 3")

# メッシュコード -> polygon
geoms = []
for m in df["meshcode"].astype(str):
    if OUTPUT_MESH_LEVEL == 3:
        west, south, east, north = meshcode3_bounds(m)
    elif OUTPUT_MESH_LEVEL == 2:
        if OUTPUT_MESH2_SUBDIV == 1:
            west, south, east, north = meshcode2_bounds(m)
        else:
            west, south, east, north = meshcode2_sub_bounds(m, subdiv=OUTPUT_MESH2_SUBDIV)
    else:
        raise ValueError("OUTPUT_MESH_LEVEL must be 2 or 3")
    geoms.append(box(west, south, east, north))

g = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")

# 面積は測地線（楕円体）で計算する（日本全国でも破綻しにくい）
from pyproj import Geod

_geod = Geod(ellps="GRS80")
if OUTPUT_MESH_LEVEL == 3:
    # 3次メッシュ（30"×45"）は緯度の行ごとに面積がほぼ同じなので、行単位でキャッシュする
    codes = g["meshcode"].astype(str)
    yy = codes.str.slice(0, 2).astype(int)
    a = codes.str.slice(4, 5).astype(int)
    c = codes.str.slice(6, 7).astype(int)
    row = (yy * 80 + a * 10 + c).astype(int)  # south_lat(deg) = row / 120

    lat_step = 30.0 / 3600.0
    lon_step = 45.0 / 3600.0
    area_by_row: dict[int, float] = {}
    for r in np.unique(row.to_numpy(dtype=np.int32)).tolist():
        south = float(r) / 120.0
        north = south + lat_step
        lons = (0.0, lon_step, lon_step, 0.0)
        lats = (south, south, north, north)
        area_by_row[int(r)] = abs(_geod.polygon_area_perimeter(lons, lats)[0]) / 1e6

    g["area_km2"] = row.map(area_by_row).astype(float)
else:
    g["area_km2"] = [
        abs(_geod.geometry_area_perimeter(geom)[0]) / 1e6 for geom in g.geometry
    ]
g["density"] = g["population"] / g["area_km2"]

# Natural Earth の日本ポリゴンが取れている場合は、海・国外のメッシュを落とす
# - 人口>0 のセルは常に残す（沿岸の誤差で落ちないように）
# - それ以外は「日本ポリゴンと交差するセル」だけ残す（海を黒で塗らない）
if "region_wgs84" in globals() and region_wgs84 is not None:
    try:
        region_wgs84 = _make_geometry_valid(region_wgs84)
        pop_pos = pd.to_numeric(g["population"], errors="coerce").fillna(-1) > 0
        on_region = g.geometry.intersects(region_wgs84)
        keep = pop_pos | on_region
        g = g.loc[keep].copy()
        print(
            "after japan clip:",
            "meshes",
            len(g),
            "missing population",
            int(pd.to_numeric(g["population"], errors="coerce").isna().sum()),
            "pop<=0",
            int((pd.to_numeric(g["population"], errors="coerce") <= 0).sum()),
        )
    except Exception as e:
        print(f"warn: japan clip failed, continue without clipping: {e}")

# Choropleth styling (shared for PNG/HTML)
g_wgs84 = g[["meshcode", "population", "area_km2", "density", "geometry"]].copy()

low_threshold = 50.0
low_color = "#2b6cb0"
no_people_color = "#000000"
fill_alpha = float(FILL_ALPHA)

dens_all = pd.to_numeric(g_wgs84["density"], errors="coerce").replace([np.inf, -np.inf], np.nan)
dens_hi = dens_all.dropna()
dens_hi = dens_hi[dens_hi > low_threshold]
if dens_hi.empty:
    vmin, vmax = low_threshold, low_threshold + 1.0
else:
    vmin = float(dens_hi.min())
    vmax = float(dens_hi.quantile(0.99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(dens_hi.min(skipna=True) or (low_threshold + 1.0))
        vmax = float(dens_hi.max(skipna=True) or (vmin + 1.0))
    if vmax <= vmin:
        vmax = vmin + 1.0

want_png = OUTPUT_MODE in ("png", "both")
want_raster_html = OUTPUT_MODE == "raster_html"
if want_png or want_raster_html:
    import json
    import os

    mpl_config_dir = Path(".cache") / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm
    from matplotlib import colors as mpl_colors
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    g_plot = g_wgs84.to_crs("EPSG:3857")
    minx, miny, maxx, maxy = g_plot.total_bounds
    dx = float(maxx - minx)
    dy = float(maxy - miny)
    if dx <= 0 or dy <= 0:
        raise RuntimeError("Invalid plot bounds for PNG output.")

    long_px = int(PNG_LONG_SIDE_PX)
    if dx >= dy:
        width_px = long_px
        height_px = max(1, int(round(long_px * (dy / dx))))
    else:
        height_px = long_px
        width_px = max(1, int(round(long_px * (dx / dy))))

    fig_w = width_px / float(PNG_DPI)
    fig_h = height_px / float(PNG_DPI)

    pop_arr = pd.to_numeric(g_wgs84["population"], errors="coerce").to_numpy(dtype=float)
    dens_arr = dens_all.to_numpy(dtype=float)

    no_people = (
        (~np.isfinite(pop_arr))
        | (pop_arr <= 0)
        | (~np.isfinite(dens_arr))
        | (dens_arr <= 0)
    )
    low = (~no_people) & (dens_arr <= low_threshold)
    high = (~no_people) & (~low)

    cmap = mpl_cm.get_cmap("YlOrRd")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    facecolors = np.empty((len(g_plot), 4), dtype=float)
    facecolors[:] = mpl_colors.to_rgba(no_people_color, alpha=fill_alpha)
    if low.any():
        facecolors[low] = mpl_colors.to_rgba(low_color, alpha=fill_alpha)
    if high.any():
        rgba = cmap(norm(dens_arr[high]))
        rgba[:, 3] = fill_alpha
        facecolors[high] = rgba

    if want_png:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PNG_DPI)
        ax.set_aspect("equal")
        ax.set_axis_off()

        g_plot.plot(ax=ax, color=facecolors, linewidth=0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        if "region_wgs84" in globals() and region_wgs84 is not None:
            try:
                gpd.GeoSeries([region_wgs84], crs="EPSG:4326").to_crs("EPSG:3857").boundary.plot(
                    ax=ax, color="#111111", linewidth=0.3, alpha=0.7
                )
            except Exception as e:
                print(f"warn: failed to plot japan outline: {e}")

        if PLOT_CITY_LABELS:
            bbox_wgs84 = tuple(g_wgs84.total_bounds.tolist())
            labels = select_city_labels(
                bbox_wgs84=bbox_wgs84,
                region_wgs84=region_wgs84 if "region_wgs84" in globals() else None,
                max_labels=int(CITY_LABEL_MAX),
                min_sep_km=float(CITY_MIN_SEP_KM),
                scale="10m",
            )
            if not labels:
                print("warn: no city labels selected; skip city labels")
            else:
                import matplotlib.patheffects as pe

                if CITY_LABEL_FONT_SIZE is None:
                    font_size = int(round(int(PNG_LONG_SIDE_PX) / 800))
                    font_size = max(8, min(14, font_size))
                else:
                    font_size = int(CITY_LABEL_FONT_SIZE)
                label_offset_m = max(4000.0, min(20000.0, max(dx, dy) / 400.0))

                xs = [float(d["x"]) for d in labels]
                ys = [float(d["y"]) for d in labels]
                ax.scatter(
                    xs,
                    ys,
                    s=float(CITY_DOT_SIZE) ** 2,
                    c="white",
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=4,
                )

                for d in labels:
                    ax.text(
                        float(d["x"]) + label_offset_m,
                        float(d["y"]),
                        str(d["name"]),
                        fontsize=font_size,
                        color="#111111",
                        ha="left",
                        va="center",
                        zorder=5,
                        path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
                    )

        title = "Japan Population Density (2020)"
        ax.set_title(title, fontsize=14, pad=12)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Population density (people/km²)")

        ax.legend(
            handles=[
                Patch(facecolor=no_people_color, edgecolor="none", label="0"),
                Patch(facecolor=low_color, edgecolor="none", label=f"1–{low_threshold:g}"),
            ],
            loc="lower left",
            frameon=True,
            framealpha=0.85,
        )

        out_path = Path(OUTPUT_PNG)
        fig.savefig(out_path, dpi=PNG_DPI)
        plt.close(fig)
        print(f"saved: {out_path} ({width_px}x{height_px}px)")

    if want_raster_html:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PNG_DPI)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_aspect("equal")
        ax.set_axis_off()

        g_plot.plot(ax=ax, color=facecolors, linewidth=0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        if "region_wgs84" in globals() and region_wgs84 is not None:
            try:
                gpd.GeoSeries([region_wgs84], crs="EPSG:4326").to_crs("EPSG:3857").boundary.plot(
                    ax=ax, color="#111111", linewidth=0.25, alpha=0.45
                )
            except Exception as e:
                print(f"warn: failed to plot japan outline on overlay: {e}")

        overlay_path = Path(OUTPUT_RASTER_PNG)
        fig.savefig(overlay_path, dpi=PNG_DPI, transparent=True)
        plt.close(fig)
        print(f"saved: {overlay_path} ({width_px}x{height_px}px)")

        minlon, minlat, maxlon, maxlat = g_wgs84.total_bounds.tolist()
        center_lat = (minlat + maxlat) / 2
        center_lon = (minlon + maxlon) / 2

        labels = []
        if PLOT_CITY_LABELS:
            labels = select_city_labels(
                bbox_wgs84=(minlon, minlat, maxlon, maxlat),
                region_wgs84=region_wgs84 if "region_wgs84" in globals() else None,
                max_labels=int(CITY_LABEL_MAX),
                min_sep_km=float(CITY_MIN_SEP_KM),
                scale="10m",
            )

        city_js = json.dumps(
            [{"name": d["name"], "lat": d["lat"], "lon": d["lon"]} for d in labels],
            ensure_ascii=False,
        )

        overlay_file = overlay_path.name
        out_html = Path(OUTPUT_RASTER_HTML)
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Japan Population Density (2020)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #map {{ height: 100%; width: 100%; }}
    .legend {{
      position: absolute;
      right: 12px;
      bottom: 12px;
      background: rgba(255,255,255,0.92);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.15);
      font: 13px/1.25 -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Arial, sans-serif;
      color: #111;
      z-index: 1000;
      max-width: 280px;
    }}
    .legend .title {{ font-weight: 600; margin-bottom: 6px; }}
    .legend .row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .legend .swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.15); }}
    .legend .grad {{ height: 10px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.15);
      background: linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #f03b20, #bd0026);
      margin-top: 6px;
    }}
    .legend .grad-labels {{ display: flex; justify-content: space-between; font-size: 12px; color: #333; }}
    .legend .small {{ font-size: 12px; color: #333; }}
    .legend input[type=range] {{ width: 100%; }}
    .leaflet-tooltip.city-label {{
      background: transparent;
      border: none;
      box-shadow: none;
      color: #111;
      font: {int(CITY_LABEL_FONT_PX)}px/1.1 -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Arial, sans-serif;
      text-shadow: 0 0 2px rgba(255,255,255,0.95), 0 0 6px rgba(255,255,255,0.85);
      padding: 0;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <div class="title">Population density (people/km²)</div>
    <div class="row"><span class="swatch" style="background:{no_people_color};"></span><span>0</span></div>
    <div class="row"><span class="swatch" style="background:{low_color};"></span><span>1–{low_threshold:g}</span></div>
    <div class="grad"></div>
    <div class="grad-labels"><span>&gt;{low_threshold:g}</span><span>{vmax:.0f} (99th pct)</span></div>
    <div class="small" style="margin-top:8px;">Overlay opacity</div>
    <input id="opacity" type="range" min="0" max="1" step="0.05" value="1" />
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const bounds = [[{minlat:.8f}, {minlon:.8f}], [{maxlat:.8f}, {maxlon:.8f}]];
    const map = L.map('map', {{ preferCanvas: true }}).setView([{center_lat:.6f}, {center_lon:.6f}], {8 if REGION == "donan" else 5});

    const base = L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{ attribution: '&copy; OpenStreetMap contributors &copy; CARTO' }}
    ).addTo(map);

    const overlay = L.imageOverlay('{overlay_file}', bounds, {{ opacity: 1 }}).addTo(map);
    map.fitBounds(bounds);

    const cities = {city_js};
    for (const c of cities) {{
      const pt = L.circleMarker([c.lat, c.lon], {{
        radius: 3,
        color: '#000',
        weight: 1,
        fillColor: '#fff',
        fillOpacity: 1
      }}).addTo(map);
      pt.bindTooltip(c.name, {{
        permanent: true,
        direction: 'right',
        offset: [8, 0],
        className: 'city-label',
        opacity: 0.9
      }});
    }}

    const opacity = document.getElementById('opacity');
    opacity.addEventListener('input', (e) => {{
      overlay.setOpacity(parseFloat(e.target.value));
    }});
  </script>
</body>
</html>
"""

        out_html.write_text(html, encoding="utf-8")
        print(f"saved: {out_html}")

if OUTPUT_MODE in ("html", "both"):
    # Folium Choropleth（人口密度をそのまま色分けしてHTML出力）
    import folium
    from branca.colormap import linear

    minx, miny, maxx, maxy = g_wgs84.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    zoom_start = 8 if REGION == "donan" else 5
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="cartodbpositron",
        prefer_canvas=True,
    )

    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = (
        f"Population density (people/km²). Black: 0, Blue: 1–{low_threshold:g}"
    )
    colormap.add_to(m)

    def style_fn(feature: dict) -> dict:
        props = feature.get("properties", {})

        pop = props.get("population")
        try:
            pop = float(pop)
        except (TypeError, ValueError):
            pop = None

        if pop is None or not np.isfinite(pop) or pop <= 0:
            return {
                "fillColor": no_people_color,
                "color": "#00000000",
                "weight": 0.0,
                "fillOpacity": fill_alpha,
            }

        v = props.get("density")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None

        if v is None or not np.isfinite(v) or v <= 0:
            return {
                "fillColor": no_people_color,
                "color": "#00000000",
                "weight": 0.0,
                "fillOpacity": fill_alpha,
            }

        if v <= low_threshold:
            return {
                "fillColor": low_color,
                "color": "#00000000",
                "weight": 0.0,
                "fillOpacity": fill_alpha,
            }

        v = max(vmin, min(vmax, v))
        return {
            "fillColor": colormap(v),
            "color": "#00000000",
            "weight": 0.0,
            "fillOpacity": fill_alpha,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["meshcode", "population", "area_km2", "density"],
        aliases=["Mesh:", "Population:", "Area (km²):", "Density (people/km²):"],
        localize=True,
        sticky=False,
    )

    folium.GeoJson(
        data=g_wgs84.to_json(),
        name="mesh choropleth",
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)

    m.fit_bounds([[miny, minx], [maxy, maxx]])
    m.save(OUTPUT_HTML)
    print(f"saved: {OUTPUT_HTML}")
