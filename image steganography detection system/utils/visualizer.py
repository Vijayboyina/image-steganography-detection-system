"""
Visualization utilities – generates matplotlib figures embedded in the GUI.
"""
from PIL import Image
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd93d"
TEXT_COL = "#e6edf3"
GRID_COL = "#30363d"


def _apply_dark_style(fig: Figure, axes):
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        if ax is None:
            continue
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        if ax.get_title():
            ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)


# ---------- Histogram chart ----------

def make_histogram_figure(analysis_results: dict) -> Figure:
    hist_data = analysis_results.get("histogram", {}).get("channels", {})
    if not hist_data:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=96)
    colors = [ACCENT, ACCENT2, ACCENT3]
    labels = list(hist_data.keys())

    for i, (name, data) in enumerate(hist_data.items()):
        ax = axes[i]
        hist = np.array(data["histogram"])
        ax.plot(hist, color=colors[i], linewidth=0.9, alpha=0.9)
        ax.fill_between(range(256), hist, alpha=0.25, color=colors[i])
        ax.set_title(f"{name} Channel", fontsize=9, color=TEXT_COL, pad=4)
        ax.set_xlabel("Pixel Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.set_xlim(0, 255)

    _apply_dark_style(fig, axes)
    fig.suptitle("Pixel-Value Histograms", color=TEXT_COL, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


# ---------- LSB planes ----------

def make_lsb_figure(np_image: np.ndarray) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), dpi=96)
    ch_names = ["Red LSB", "Green LSB", "Blue LSB"]
    colors = [ACCENT, ACCENT2, ACCENT3]

    for i, (name, col) in enumerate(zip(ch_names, colors)):
        ax = axes[i]
        lsb_plane = (np_image[:, :, i] & 1) * 255
        ax.imshow(lsb_plane, cmap="gray",
                  interpolation="nearest", aspect="auto")
        ax.set_title(name, fontsize=9, color=TEXT_COL, pad=4)
        ax.axis("off")

    _apply_dark_style(fig, axes)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("LSB Planes (white = bit 1)",
                 color=TEXT_COL, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig


# ---------- Frequency spectrum ----------

def make_fft_figure(analysis_results: dict) -> Figure:
    mag = analysis_results.get("dct", {}).get("magnitude_spectrum")
    if mag is None:
        return None

    mag = np.array(mag)
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=96)
    im = ax.imshow(mag, cmap="inferno",
                   interpolation="bilinear", aspect="auto")
    ax.set_title("Log Magnitude Spectrum (FFT)",
                 fontsize=9, color=TEXT_COL, pad=4)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.outline.set_edgecolor(GRID_COL)

    _apply_dark_style(fig, [ax])
    fig.tight_layout()
    return fig


# ---------- Risk radar / bar ----------

def make_risk_figure(overall: dict) -> Figure:
    detections = overall.get("detections", {})
    if not detections:
        return None

    labels = list(detections.keys())
    values = [1 if v else 0 for v in detections.values()]
    colors = [ACCENT2 if v else ACCENT for v in values]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=96)
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean", "Detected"], fontsize=8)
    ax.set_title("Detection Results per Test",
                 fontsize=9, color=TEXT_COL, pad=6)

    for bar, val in zip(bars, values):
        label = "DETECTED" if val else "CLEAN"
        ax.text(
            val + 0.03, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=7,
            color=ACCENT2 if val else ACCENT,
        )

    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


# ---------- figure → PIL Image ----------

def fig_to_pil(fig: Figure) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img
