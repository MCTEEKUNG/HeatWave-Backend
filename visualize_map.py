"""
Heatwave AI — OSM Map Visualization
Renders Bangkok heatwave risk level on an OpenStreetMap basemap.
Outputs a static PNG image.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import requests
import math
import os
import io
import config
from datetime import datetime


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """Convert tile numbers back to lat/lon."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def fetch_osm_tiles(bbox, zoom=14):
    """Fetch OSM tiles covering the bounding box and stitch them."""
    x_min, y_min = deg2num(bbox["north"], bbox["west"], zoom)
    x_max, y_max = deg2num(bbox["south"], bbox["east"], zoom)

    x_start = int(math.floor(x_min))
    x_end = int(math.floor(x_max))
    y_start = int(math.floor(y_min))
    y_end = int(math.floor(y_max))

    tile_size = 256
    width = (x_end - x_start + 1) * tile_size
    height = (y_end - y_start + 1) * tile_size

    result = Image.new("RGBA", (width, height))

    print(f"  Fetching OSM tiles: zoom={zoom}, "
          f"{(x_end - x_start + 1) * (y_end - y_start + 1)} tiles...")

    headers = {"User-Agent": "HeatwaveAI/1.0 (research project)"}

    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                px = (x - x_start) * tile_size
                py = (y - y_start) * tile_size
                result.paste(tile, (px, py))
            except Exception as e:
                print(f"    Warning: Failed to fetch tile {x},{y}: {e}")

    # Calculate pixel coordinates of the bbox within the stitched image
    px_west = (x_min - x_start) * tile_size
    px_east = (x_max - x_start) * tile_size
    py_north = (y_min - y_start) * tile_size
    py_south = (y_max - y_start) * tile_size

    tile_bounds = {
        "px_west": px_west, "px_east": px_east,
        "py_north": py_north, "py_south": py_south,
        "x_start": x_start, "y_start": y_start,
        "zoom": zoom,
    }

    return result, tile_bounds


def get_risk_color(risk_level):
    """Get RGBA color for risk level."""
    colors = config.RISK_COLORS
    level = risk_level.upper().replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "").strip()
    return colors.get(level, colors["LOW"])


def render_heatwave_map(risk_level, probability, weather_data, date_str=None,
                        output_path=None):
    """
    Render Bangkok heatwave risk map on OSM basemap.
    
    Args:
        risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        probability: float 0-1
        weather_data: dict with T2M_MAX, PRECTOTCORR, WS10M, RH2M, NDVI
        date_str: date string for title
        output_path: where to save the PNG
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if output_path is None:
        output_path = os.path.join(config.MAPS_DIR, f"heatwave_map_{date_str}.png")

    bbox = config.BBOX

    # Fetch OSM tiles
    print("🗺️  Generating heatwave risk map...")
    osm_img, bounds = fetch_osm_tiles(bbox, zoom=14)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(np.array(osm_img))

    # Draw risk overlay
    color = get_risk_color(risk_level)
    overlay_color = (color[0]/255, color[1]/255, color[2]/255, color[3]/255)

    rect = plt.Rectangle(
        (bounds["px_west"], bounds["py_north"]),
        bounds["px_east"] - bounds["px_west"],
        bounds["py_south"] - bounds["py_north"],
        facecolor=overlay_color,
        edgecolor=(overlay_color[0], overlay_color[1], overlay_color[2], 0.9),
        linewidth=2.5,
    )
    ax.add_patch(rect)

    # Center marker
    cx = (bounds["px_west"] + bounds["px_east"]) / 2
    cy = (bounds["py_north"] + bounds["py_south"]) / 2
    ax.plot(cx, cy, "o", markersize=10, color="white", markeredgecolor="black",
            markeredgewidth=1.5, zorder=5)

    # Risk level icons mapping
    icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
    icon = icons.get(risk_level, "⚪")

    # Title
    ax.set_title(
        f"Heatwave Risk Map — Bangkok ({date_str})\n"
        f"{icon} {risk_level} | Probability: {probability:.1%}",
        fontsize=16, fontweight="bold", pad=15,
    )

    # Info box (bottom-left)
    info_lines = [
        f"📅 Date: {date_str}",
        f"🌡️ Max Temp: {weather_data.get('T2M_MAX', 'N/A'):.1f}°C",
        f"💧 Rainfall: {weather_data.get('PRECTOTCORR', 'N/A'):.1f} mm",
        f"💨 Wind: {weather_data.get('WS10M', 'N/A'):.1f} m/s",
        f"💦 Humidity: {weather_data.get('RH2M', 'N/A'):.1f}%",
        f"🌿 NDVI: {weather_data.get('NDVI', 'N/A'):.4f}",
    ]
    info_text = "\n".join(info_lines)

    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
    ax.text(
        0.02, 0.02, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="bottom",
        bbox=props, family="monospace",
    )

    # Legend (bottom-right)
    legend_entries = [
        mpatches.Patch(facecolor=(0, 180/255, 0, 0.25), edgecolor="green", label="🟢 LOW (< 40%)"),
        mpatches.Patch(facecolor=(1, 200/255, 0, 0.35), edgecolor="orange", label="🟡 MEDIUM (40-60%)"),
        mpatches.Patch(facecolor=(1, 120/255, 0, 0.45), edgecolor="darkorange", label="🟠 HIGH (60-80%)"),
        mpatches.Patch(facecolor=(220/255, 0, 0, 0.55), edgecolor="red", label="🔴 CRITICAL (> 80%)"),
    ]
    ax.legend(handles=legend_entries, loc="lower right", fontsize=9,
              fancybox=True, framealpha=0.85, edgecolor="gray")

    # Attribution
    ax.text(
        0.99, 1.01, "© OpenStreetMap contributors",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color="gray",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"  Map saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Demo: render a sample map
    demo_weather = {
        "T2M_MAX": 33.4,
        "PRECTOTCORR": 1.1,
        "WS10M": 1.5,
        "RH2M": 77.3,
        "NDVI": 0.4244,
    }
    render_heatwave_map(
        risk_level="LOW",
        probability=0.0,
        weather_data=demo_weather,
        date_str="2026-02-16",
    )
