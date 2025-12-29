from PIL import Image, ImageDraw, ImageFont
from configparser import ConfigParser
from openai import OpenAI
import numpy as np
from PIL import Image
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def crop_to_image(slide, x0, y0, w, h, target_max=1000, oversample_bias=0.85):
    """
    Crop a region from a whole-slide image and return a Pillow Image
    that is ~target_max on its long side, with high-quality downsampling.
    - oversample_bias < 1.0 biases toward a finer (higher-res) level to avoid upsampling.
    """
    # Sanity clamps (level-0 space)
    W0, H0 = slide.level_dimensions[0]
    x0 = max(0, min(x0, W0))
    y0 = max(0, min(y0, H0))
    w  = max(1, min(w, W0 - x0))
    h  = max(1, min(h, H0 - y0))

    # Desired overall downsample to hit target_max
    long_side = max(w, h)
    if long_side <= 0:
        raise ValueError("Empty crop region.")

    desired_down = long_side / float(target_max)

    # Use OpenSlide's chooser; bias to a finer level so we downscale (not upsample)
    biased_down = max(1.0, desired_down * oversample_bias)
    lvl = slide.get_best_level_for_downsample(biased_down)

    # If that level would still produce a smaller-than-target image, try one level finer
    # to avoid blurry upsampling.
    down_at_lvl = float(slide.level_downsamples[lvl])
    rend_w = max(1, int(round(w / down_at_lvl)))
    rend_h = max(1, int(round(h / down_at_lvl)))
    if max(rend_w, rend_h) < target_max and lvl > 0:
        lvl = lvl - 1
        down_at_lvl = float(slide.level_downsamples[lvl])
        rend_w = max(1, int(round(w / down_at_lvl)))
        rend_h = max(1, int(round(h / down_at_lvl)))

    # Read region at that level (coords are still in level-0 space)
    region = slide.read_region((int(x0), int(y0)), lvl, (rend_w, rend_h)).convert("RGB")

    # Now scale to exactly target_max on the long side (downscale only; avoid upscaling big jumps)
    long_r = max(region.width, region.height)
    if long_r > target_max:
        scale = target_max / float(long_r)
        new_w = max(1, int(round(region.width * scale)))
        new_h = max(1, int(round(region.height * scale)))
        region = region.resize((new_w, new_h), Image.LANCZOS)

    return region


def add_position_guides(
    region,
    *,
    x0: int,
    y0: int,
    down: int,
    n_vertical: int = 4,
    n_horizontal: int = 4,
    show_edges: bool = True,
    font_path: str | None = None,
    font_size: int = 14,
):
    img   = region.convert("RGBA")
    draw  = ImageDraw.Draw(img, "RGBA")
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # Use DejaVuSans.ttf as the default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            # If DejaVuSans.ttf not found, use default (but warn that size is ignored)
            font = ImageFont.load_default()
            print(f"Warning: DejaVuSans.ttf not found, using default font, font_size={font_size} ignored")
    
    w, h  = img.size
    w_full, h_full = w * down, h * down     # region size in full-res px

    # --- Pillow-version-safe text measure ----------------------------------
    def measure(text: str) -> tuple[int, int]:
        """Return (width, height) of *text* for current font."""
        try:                                        # Pillow ≥ 8.0
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t
        except AttributeError:                      # Pillow < 8.0
            return draw.textsize(text, font=font)
    
    def get_text_bbox(text: str, pos: tuple) -> tuple:
        """Get accurate text bounding box for positioning."""
        try:                                        # Pillow ≥ 8.0
            return draw.textbbox(pos, text, font=font)
        except AttributeError:                      # Pillow < 8.0
            tw, th = draw.textsize(text, font=font)
            return (pos[0], pos[1], pos[0] + tw, pos[1] + th)

    # --- small helper to draw a labelled guide line ------------------------
    def _line(p1, p2, label, flip=False):
        draw.line([p1, p2], fill=(0, 255, 0, 180), width=1)
        
        # Position text near the line start point
        if not flip:  # Vertical lines - text at top
            tx, ty = p1[0] + 3, p1[1] + 3
        else:  # Horizontal lines - text positioned above the line
            # Get text dimensions first to position properly above line
            tw, th = measure(label)
            tx, ty = p1[0] + 3, p1[1] - th - 6  # 6px clearance above line
        
        # Get accurate text bounding box
        bbox = get_text_bbox(label, (tx, ty))
        l, t, r, b = bbox
        
        # Add padding around the text box
        padding = 3
        draw.rectangle([l - padding, t - padding, r + padding, b + padding],
                       fill=(0, 0, 0, 150))
        draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

    # Vertical guides
    for i in range(1, n_vertical + 1):
        frac = i / (n_vertical + 1)
        sx   = frac * w
        fx   = x0 + int(frac * w_full)            # full-res X
        _line((sx, 0), (sx, h), str(fx), flip=False)

    # Horizontal guides
    for j in range(1, n_horizontal + 1):
        frac = j / (n_horizontal + 1)
        sy   = frac * h
        fy   = y0 + int(frac * h_full)            # full-res Y
        _line((0, sy), (w, sy), str(fy), flip=True)

    # Optional border
    if show_edges:
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 0, 0, 180), width=1)
        corner_lbl = f"({x0}, {y0})"
        tx, ty = (2, 2)
        
        # Get accurate text bounding box for corner label
        bbox = get_text_bbox(corner_lbl, (tx, ty))
        l, t, r, b = bbox
        
        # Add padding around corner label
        padding = 4
        draw.rectangle([l - padding, t - padding, r + padding, b + padding], 
                       fill=(0, 0, 0, 150))
        draw.text((tx, ty), corner_lbl, font=font, fill=(255, 255, 255, 255))

    return img


def crop_to_image_with_guides(slide, x0, y0, w, h, target_max=1000, oversample_bias=0.85):
    img = crop_to_image(slide, x0, y0, w, h, target_max=target_max, oversample_bias=oversample_bias)

    # Scale font by rendered size, not by pyramid math (more robust)
    target_font_density = 0.02
    min_dim = min(img.width, img.height)
    scaled_font_size = max(int(target_font_density * min_dim), 8)

    # Use the *actual* effective downsample used to render (approximate)
    # This helps draw correct coordinate ticks if your guide function needs it.
    # effective_down ≈ w / img.width, but use long sides for stability.
    eff_down = max(w, h) / float(max(img.width, img.height))

    region_with_guides = add_position_guides(
        img,
        x0=x0, y0=y0, down=eff_down,
        n_vertical=4, n_horizontal=4,
        font_path=None, font_size=scaled_font_size
    ).convert('RGB')

    return region_with_guides