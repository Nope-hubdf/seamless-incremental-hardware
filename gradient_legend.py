from typing import NamedTuple, Literal

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import BboxImage
import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox, Bbox
import numpy as np

class GradientHandle(NamedTuple):
    rgb_src: tuple[float, float, float] | tuple[float, float, float, float]
    """Source color (RGB(A) in [0., 1.])"""
    rgb_dst: tuple[float, float, float] | tuple[float, float, float, float]
    """Destination color (RGB(A) in [0., 1.])"""
    approach: Literal["interpolate", "linspace"]
    """
    Interpolation approach (affects appearance): create 2-element image, then interpolate color with bicubic
    interpolation (``interpolate``); create ``num``-element linspace image, then use nearest colors (``linspace``)
    """
    num: int = 1000
    """Number of steps ("quantization levels") in colormap"""
    
    @property
    def cm(self) -> LinearSegmentedColormap:
        """:return: Colormap from ``rgb_src`` to ``rgb_dst`` with ``num`` steps"""
        # https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#linearsegmentedcolormap
        return LinearSegmentedColormap.from_list(f"LSC:{self}", [self.rgb_src, self.rgb_dst], N=self.num)
        
class GradientHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        # Place tiny gradient image (https://stackoverflow.com/a/42971319/7395592) into legend
        # (https://stackoverflow.com/a/32303541/7395592), apply colormap
        bounds = (b := handlebox).xdescent, b.ydescent, b.width, b.height
        bbox = TransformedBbox(Bbox.from_bounds(*bounds), transform=b.get_transform())
        if orig_handle.approach == "interpolate":
            interpolation, image_data = "bicubic", [[0, 1]]
        elif orig_handle.approach == "linspace":
            interpolation, image_data = "nearest", np.linspace(0, 1, num=orig_handle.num)[np.newaxis]
        else:
            raise ValueError(f"Unknown interpolation approach '{orig_handle.approach}'")
        img = BboxImage(bbox, cmap=orig_handle.cm, interpolation=interpolation)
        img.set_data(image_data)
        handlebox.add_artist(img)
        return img

if __name__ == "__main__":
    handle = GradientHandle(rgb_src=(1, 0, 0), rgb_dst=(0, 0, 1), approach="linspace")  # Red → blue
    plt.legend([handle], ["Look, a gradient!"], handler_map={GradientHandle: GradientHandler()})
