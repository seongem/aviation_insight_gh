import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class PaletteManager:
    """Centralized Okabe-Ito palette, layout constants, and typography settings

    for praxis/publication figures.
    """

    # Okabe-Ito Colorblind-Friendly Palette (8-color standard)
    OKABE_ITO = [
        "#0072B2",  # 0: Blue (Primary)
        "#E69F00",  # 1: Orange
        "#009E73",  # 2: Bluish Green
        "#56B4E9",  # 3: Sky Blue
        "#D55E00",  # 4: Vermilion
        "#CC79A7",  # 5: Reddish Purple
        "#F0E442",  # 6: Yellow
        "#000000",  # 7: Black
    ]

    # Distinct Accent for Threshold / Anomaly Overlays
    CRIMSON = "#B22222"  # Firebrick Crimson for E(t) overlays & anomaly fills

    # Typography & Axis Colors for High-Contrast Publication Graphics
    TEXT_COLOR = "#000000"       # General text & tick labels
    LABEL_COLOR = "#000000"      # Axis titles (xlabel / ylabel)
    TITLE_COLOR = "#000000"      # Plot / Subplot headers
    GRID_COLOR = "#E0E0E0"       # Subtle gridlines

    # Layout & Page Geometry (Optimized for 8.5" x 11" Portrait / Quarto)
    PAGE_WIDTH_INCHES = 7.0  # Printable width with 0.75" margins
    DPI = 300

    def __init__(self):
        # Pre-configured continuous colormaps for heatmaps/confusion matrices
        self.cmap_blue = LinearSegmentedColormap.from_list(
            "okabe_blue", ["#FAFAFA", "#0072B2"]
        )
        self.cmap_vermilion = LinearSegmentedColormap.from_list(
            "okabe_vermilion", ["#FAFAFA", "#D55E00"]
        )

    @classmethod
    def get_sensor_color(cls, idx: int) -> str:
        """Returns a deterministic Okabe-Ito hex color based on channel index."""
        return cls.OKABE_ITO[idx % len(cls.OKABE_ITO)]

    @classmethod
    def apply_publication_style(cls, style="seaborn-v0_8-whitegrid"):
        """Sets standard matplotlibrc context for clean Quarto PDF/Word exports."""
        plt.style.use(style)
        plt.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "savefig.dpi": cls.DPI,
            "figure.autolayout": False,
            
            # Global Typography High-Contrast Color Overrides
            "text.color": cls.TEXT_COLOR,
            "axes.labelcolor": cls.LABEL_COLOR,
            "axes.titlecolor": cls.TITLE_COLOR,
            "xtick.color": cls.TEXT_COLOR,
            "ytick.color": cls.TEXT_COLOR,
            "axes.edgecolor": cls.TEXT_COLOR,
            "grid.color": cls.GRID_COLOR,
        })


# Global singleton instance for quick importing
theme = PaletteManager()