# config.py

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
PREFERRED_DEVICE_ORDER = ["mps", "cpu"]

# File types we will scan
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Where we store the reference index
INDEX_DIR = "output/index"
REFERENCE_INDEX_FILE = "output/index/photosymbols_index.npz"
REFERENCE_META_FILE = "output/index/photosymbols_meta.csv"

# --- scanning settings ---

# output
REPORT_DIR = "output/report"
THUMBS_DIR = "output/report/thumbs"
REPORT_CSV = "output/report/report.csv"
REPORT_HTML = "output/report/report.html"

# Thumbnail size for previews
THUMB_SIZE = (220, 220)

# Make CLIP much stricter
CONFIRMED_THRESHOLD = 0.55
REVIEW_THRESHOLD = 0.45

# Require "best match is clearly better than runner-up"
MIN_MARGIN = 0.06

# Downscale huge images for speed/safety
MAX_DIM = 2048

# ORB verification (simple feature match)
ORB_MIN_MATCHES = 20

# Crop strategy for flattened images (safety-first but not insane)
TILE_SIZES = [256, 384]     # pixels
TILE_STRIDE = 192           # pixels (overlap to catch partial people)
MAX_TILES_PER_IMAGE = 80    # safety cap so one giant image doesn't explode runtime

# --- strict matching upgrades ---

# Only use CLIP to shortlist top K reference candidates
TOPK_REFS = 10

# ORB verification settings (stricter = fewer false positives)
ORB_MIN_MATCHES_REVIEW = 25
ORB_MIN_MATCHES_CONFIRMED = 35

# Resize safety (already using MAX_DIM, keep it)
# MAX_DIM = 2048

# Report filtering
INCLUDE_CLEAR_IN_REPORT = False

# --- PDF scanning ---
PDF_RENDER_ZOOM = 2.0       # 2.0 is a good “safe but slower” default
PDF_MAX_TILES_PER_PAGE = 160  # pages can be large; cap to keep runtime sane
PDF_OUTPUT_PREFIX = "pdf"     # keeps outputs grouped
