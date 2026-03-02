# scanner.py
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip
from tqdm import tqdm

import cv2
import fitz  # PyMuPDF

from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    PREFERRED_DEVICE_ORDER,
    IMAGE_EXTS,
    INDEX_DIR,
    REFERENCE_INDEX_FILE,
    REFERENCE_META_FILE,
    REPORT_DIR,
    THUMBS_DIR,
    REPORT_CSV,
    REPORT_HTML,
    THUMB_SIZE,
    TILE_SIZES,
    TILE_STRIDE,
    MAX_TILES_PER_IMAGE,
    MIN_MARGIN,
    MAX_DIM,
    TOPK_REFS,
    ORB_MIN_MATCHES_REVIEW,
    ORB_MIN_MATCHES_CONFIRMED,
    INCLUDE_CLEAR_IN_REPORT,
    # PDF
    PDF_RENDER_ZOOM,
    PDF_MAX_TILES_PER_PAGE,
    PDF_OUTPUT_PREFIX,
)


# ---------------- basics ----------------

def pick_device() -> str:
    for d in PREFERRED_DEVICE_ORDER:
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
        if d == "cpu":
            return "cpu"
    return "cpu"


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess


def iter_images(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)


# ---------------- safety resize ----------------

def shrink(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def safe_open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


# ---------------- embeddings / matching ----------------

def embed_pil(img: Image.Image, model, preprocess, device: str) -> np.ndarray:
    img = shrink(img, MAX_DIM).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


def topk_indices(query_emb: np.ndarray, ref_embs: np.ndarray, k: int):
    scores = ref_embs @ query_emb
    if scores.shape[0] <= k:
        idxs = np.argsort(scores)[::-1]
        return idxs.tolist(), scores[idxs].tolist()
    idxs = np.argpartition(scores, -k)[-k:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    return idxs.tolist(), scores[idxs].tolist()


def load_reference_index():
    if not Path(REFERENCE_INDEX_FILE).exists() or not Path(REFERENCE_META_FILE).exists():
        raise FileNotFoundError(
            "Reference index not found. Build it first:\n"
            '  python scanner.py index "/path/to/PhotoSymbols"\n'
        )
    data = np.load(REFERENCE_INDEX_FILE)
    ref_embs = data["embeddings"].astype(np.float32)
    ref_meta = pd.read_csv(REFERENCE_META_FILE)
    return ref_embs, ref_meta


# ---------------- crops / tiles ----------------

def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or ("transparency" in img.info)


def tile_generator(img: Image.Image, max_tiles: int):
    img = shrink(img, MAX_DIM).convert("RGB")
    w, h = img.size
    tiles_yielded = 0

    for tile_size in TILE_SIZES:
        if tile_size > w and tile_size > h:
            continue

        for top in range(0, max(1, h - tile_size + 1), TILE_STRIDE):
            for left in range(0, max(1, w - tile_size + 1), TILE_STRIDE):
                crop = img.crop((left, top, left + tile_size, top + tile_size))
                yield crop, (left, top, tile_size, tile_size)
                tiles_yielded += 1
                if tiles_yielded >= max_tiles:
                    return


def alpha_blob_crops(img: Image.Image):
    img = shrink(img, MAX_DIM)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[-1])
    mask = alpha > 10
    if mask.sum() < 50:
        return []

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.size[0] - 1, x2 + pad)
    y2 = min(img.size[1] - 1, y2 + pad)

    crop = img.crop((x1, y1, x2 + 1, y2 + 1)).convert("RGB")
    return [(crop, (x1, y1, x2 - x1 + 1, y2 - y1 + 1))]


# ---------------- ORB verification ----------------

def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    img = shrink(img, MAX_DIM).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def orb_match_count(a_gray: np.ndarray, b_gray: np.ndarray) -> int:
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(a_gray, None)
    kp2, des2 = orb.detectAndCompute(b_gray, None)
    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = 0
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good += 1
    return good


def verify_same_asset(query_img: Image.Image, ref_img: Image.Image) -> int:
    q = pil_to_gray_np(query_img)
    r = pil_to_gray_np(ref_img)
    return orb_match_count(q, r)


# ---------------- thumbnails + report ----------------

def save_thumb(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb = shrink(img, 1200).convert("RGB")
    thumb.thumbnail(THUMB_SIZE)
    thumb.save(out_path, format="JPEG", quality=85)


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def write_html_report_simple(rows, title: str):
    """
    SIMPLE TEAM REPORT:
      Status | Page | Target | Matched
    """
    lines = []
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append("<style>")
    lines.append("body{font-family:Arial, sans-serif;}")
    lines.append("table{border-collapse:collapse; width:100%;}")
    lines.append("th,td{border:1px solid #ddd; padding:8px; vertical-align:top;}")
    lines.append("th{background:#f4f4f4;}")
    lines.append(".tag{padding:2px 6px; border-radius:6px; font-weight:bold; display:inline-block;}")
    lines.append(".CONFIRMED{background:#ffcccc;}")
    lines.append(".REVIEW{background:#fff2cc;}")
    lines.append(".ERROR{background:#e0e0e0;}")
    lines.append("</style></head><body>")
    lines.append(f"<h2>{html_escape(title)}</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Status</th><th>Page</th><th>Target</th><th>Matched PhotoSymbols</th></tr>")

    for r in rows:
        status = r["status"]
        page = r.get("page", "")
        target_thumb = r.get("target_thumb", "")
        ref_thumb = r.get("ref_thumb", "")

        lines.append("<tr>")
        lines.append(f"<td><span class='tag {status}'>{status}</span></td>")
        lines.append(f"<td>{html_escape(str(page))}</td>")
        lines.append(f"<td><img src='{html_escape(target_thumb)}'></td>" if target_thumb else "<td>(no image)</td>")
        lines.append(f"<td><img src='{html_escape(ref_thumb)}'></td>" if ref_thumb else "<td>(none)</td>")
        lines.append("</tr>")

    lines.append("</table></body></html>")
    Path(REPORT_HTML).write_text("\n".join(lines), encoding="utf-8")


# ---------------- COMMAND: index ----------------

def build_reference_index(photosymbols_folder: Path):
    ensure_dirs()

    device = pick_device()
    print(f"Using device: {device}")
    model, preprocess = load_model(device)
    print("✅ Model loaded")

    paths = list(iter_images(photosymbols_folder))
    if not paths:
        print("❌ No images found in PhotoSymbols folder.")
        return 1

    print(f"Found {len(paths)} PhotoSymbols images. Building index...")

    embeddings = []
    meta_rows = []
    failed = []

    for p in tqdm(paths, desc="Embedding PhotoSymbols"):
        try:
            img = safe_open_image(p)
            emb = embed_pil(img, model, preprocess, device)
            embeddings.append(emb)
            meta_rows.append({"ref_path": str(p), "ref_name": p.name, "ref_ext": p.suffix.lower()})
        except Exception as e:
            failed.append({"ref_path": str(p), "ref_name": p.name, "error": str(e)})

    if not embeddings:
        print("❌ No embeddings succeeded.")
        return 1

    emb_array = np.vstack(embeddings).astype(np.float32)

    os.makedirs(Path(REFERENCE_INDEX_FILE).parent, exist_ok=True)
    np.savez_compressed(REFERENCE_INDEX_FILE, embeddings=emb_array)
    pd.DataFrame(meta_rows).to_csv(REFERENCE_META_FILE, index=False)

    if failed:
        fail_path = Path(REFERENCE_META_FILE).with_name("photosymbols_failed.csv")
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        print(f"⚠️ {len(failed)} PhotoSymbols files failed to embed. Saved:", fail_path)

    print("✅ Saved:")
    print(" -", REFERENCE_INDEX_FILE)
    print(" -", REFERENCE_META_FILE)
    print("Embeddings shape:", emb_array.shape)
    return 0


# ---------------- shared scan helpers ----------------

def decide_status(clip_top1: float, clip_top2: float, orb_matches: int) -> str:
    margin = clip_top1 - clip_top2
    if margin < MIN_MARGIN:
        return "CLEAR"
    if orb_matches >= ORB_MIN_MATCHES_CONFIRMED:
        return "CONFIRMED"
    if orb_matches >= ORB_MIN_MATCHES_REVIEW:
        return "REVIEW"
    return "CLEAR"


def scan_candidate(query_img: Image.Image, ref_paths: list[str]) -> tuple[str, int]:
    best_ref = ""
    best_matches = 0
    for rp in ref_paths:
        try:
            ref_img = safe_open_image(Path(rp))
            m = verify_same_asset(query_img, ref_img)
            if m > best_matches:
                best_matches = m
                best_ref = rp
        except Exception:
            continue
    return best_ref, best_matches


def scan_pil_image(query_img: Image.Image, model, preprocess, device: str, ref_embs, ref_meta, max_tiles: int):
    """
    Runs: CLIP shortlist -> ORB verify, on:
      - whole image
      - alpha crop OR tiles
    Returns dict with status/ref_path/method.
    """
    # Whole -> shortlist
    whole_emb = embed_pil(query_img, model, preprocess, device)
    idxs, scores = topk_indices(whole_emb, ref_embs, TOPK_REFS)
    clip1 = float(scores[0]) if scores else float("-inf")
    clip2 = float(scores[1]) if len(scores) > 1 else float("-inf")

    best = {"status": "CLEAR", "ref_path": "", "orb": 0, "clip1": clip1, "clip2": clip2, "method": "whole"}

    def try_one(img_part: Image.Image, idxs_local, scores_local, method_name: str):
        nonlocal best
        c1 = float(scores_local[0]) if scores_local else float("-inf")
        c2 = float(scores_local[1]) if len(scores_local) > 1 else float("-inf")
        if (c1 - c2) < MIN_MARGIN:
            return

        ref_paths = [str(ref_meta.iloc[i]["ref_path"]) for i in idxs_local]
        ref_path, orb_m = scan_candidate(img_part, ref_paths)
        status = decide_status(c1, c2, orb_m)

        # prefer higher orb, then higher clip
        if orb_m > best["orb"] or (orb_m == best["orb"] and c1 > best["clip1"]):
            best = {"status": status, "ref_path": ref_path if status != "CLEAR" else "", "orb": orb_m, "clip1": c1, "clip2": c2, "method": method_name}

    # try whole
    try_one(query_img.convert("RGB"), idxs, scores, "whole")

    # alpha blobs or tiles
    crops = []
    try:
        if has_alpha(query_img):
            crops = alpha_blob_crops(query_img)
    except Exception:
        crops = []

    if crops:
        for crop, _bbox in crops:
            emb = embed_pil(crop, model, preprocess, device)
            idxs2, scores2 = topk_indices(emb, ref_embs, TOPK_REFS)
            try_one(crop, idxs2, scores2, "alpha_crop")
    else:
        for crop, _bbox in tile_generator(query_img.convert("RGB"), max_tiles=max_tiles):
            emb = embed_pil(crop, model, preprocess, device)
            idxs2, scores2 = topk_indices(emb, ref_embs, TOPK_REFS)
            try_one(crop, idxs2, scores2, "tiles")

    return best


# ---------------- COMMAND: scan (folder) ----------------

def run_scan(photos_folder: Path):
    ensure_dirs()

    ref_embs, ref_meta = load_reference_index()
    print(f"Loaded PhotoSymbols reference index: {len(ref_meta)} images")

    device = pick_device()
    print(f"Using device: {device}")
    model, preprocess = load_model(device)
    print("✅ Model loaded")

    targets = list(iter_images(photos_folder))
    if not targets:
        print("❌ No images found in target folder.")
        return 1

    print(f"Found {len(targets)} target images. Scanning...")

    rows = []
    for p in tqdm(targets, desc="Scanning targets"):
        try:
            img = safe_open_image(p)
            result = scan_pil_image(img, model, preprocess, device, ref_embs, ref_meta, max_tiles=MAX_TILES_PER_IMAGE)
            rows.append({
                "target_path": str(p),
                "status": result["status"],
                "ref_path": result["ref_path"],
                "method": result["method"],
                "orb_matches": result["orb"],
            })
        except Exception as e:
            rows.append({"target_path": str(p), "status": "ERROR", "ref_path": "", "method": "error", "orb_matches": 0, "error": str(e)})

    df = pd.DataFrame(rows)
    if not INCLUDE_CLEAR_IN_REPORT:
        df_out = df[df["status"].isin(["CONFIRMED", "REVIEW", "ERROR"])].copy()
    else:
        df_out = df.copy()

    df_out.to_csv(REPORT_CSV, index=False)

    # simple html (no clear)
    html_rows = []
    for i, r in enumerate(df_out.to_dict(orient="records")):
        try:
            target_img = safe_open_image(Path(r["target_path"]))
            target_thumb_path = Path(THUMBS_DIR) / "targets" / f"{i:05d}.jpg"
            save_thumb(target_img, target_thumb_path)

            ref_thumb_path_str = ""
            if r["status"] in ("CONFIRMED", "REVIEW") and r["ref_path"]:
                ref_img = safe_open_image(Path(r["ref_path"]))
                ref_thumb_path = Path(THUMBS_DIR) / "refs" / f"{i:05d}.jpg"
                save_thumb(ref_img, ref_thumb_path)
                ref_thumb_path_str = str(ref_thumb_path)

            target_rel = os.path.relpath(target_thumb_path, Path(REPORT_HTML).parent)
            ref_rel = os.path.relpath(ref_thumb_path_str, Path(REPORT_HTML).parent) if ref_thumb_path_str else ""

            html_rows.append({"status": r["status"], "page": "", "target_thumb": target_rel, "ref_thumb": ref_rel})
        except Exception:
            continue

    write_html_report_simple(html_rows, title="PhotoSymbols Folder Scan (Flagged Only)")

    print("✅ Outputs created:")
    print(" -", REPORT_CSV)
    print(" -", REPORT_HTML)
    print(" - thumbnails in", THUMBS_DIR)
    return 0


# ---------------- COMMAND: scan_pdf ----------------

def render_pdf_page(doc: fitz.Document, page_number_1based: int) -> Image.Image:
    page = doc.load_page(page_number_1based - 1)
    mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def run_scan_pdf(pdf_path: Path):
    ensure_dirs()

    ref_embs, ref_meta = load_reference_index()
    print(f"Loaded PhotoSymbols reference index: {len(ref_meta)} images")

    device = pick_device()
    print(f"Using device: {device}")
    model, preprocess = load_model(device)
    print("✅ Model loaded")

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    print(f"Scanning PDF: {pdf_path.name} ({total_pages} pages) — render+tiles (safe but slower)")

    flagged = []

    # We’ll make unique thumbs per page
    thumb_base = Path(THUMBS_DIR) / PDF_OUTPUT_PREFIX
    thumb_targets = thumb_base / "targets"
    thumb_refs = thumb_base / "refs"

    for page_no in tqdm(range(1, total_pages + 1), desc="Scanning pages"):
        try:
            page_img = render_pdf_page(doc, page_no)

            # Scan the rendered page using a higher tile cap for safety
            result = scan_pil_image(
                page_img, model, preprocess, device, ref_embs, ref_meta,
                max_tiles=PDF_MAX_TILES_PER_PAGE
            )

            if result["status"] in ("CONFIRMED", "REVIEW"):
                # Save thumbs
                target_thumb_path = thumb_targets / f"page_{page_no:04d}.jpg"
                save_thumb(page_img, target_thumb_path)

                ref_thumb_path_str = ""
                if result["ref_path"]:
                    ref_img = safe_open_image(Path(result["ref_path"]))
                    ref_thumb_path = thumb_refs / f"page_{page_no:04d}.jpg"
                    save_thumb(ref_img, ref_thumb_path)
                    ref_thumb_path_str = str(ref_thumb_path)

                # Make paths relative to report.html location
                target_rel = os.path.relpath(target_thumb_path, Path(REPORT_HTML).parent)
                ref_rel = os.path.relpath(ref_thumb_path_str, Path(REPORT_HTML).parent) if ref_thumb_path_str else ""

                flagged.append({
                    "status": result["status"],
                    "page": page_no,
                    "target_thumb": target_rel,
                    "ref_thumb": ref_rel,
                    "target": f"{pdf_path.name}#page={page_no}",
                    "ref_path": result["ref_path"],
                    "method": result["method"],
                    "orb_matches": result["orb"],
                })

        except Exception as e:
            flagged.append({
                "status": "ERROR",
                "page": page_no,
                "target_thumb": "",
                "ref_thumb": "",
                "target": f"{pdf_path.name}#page={page_no}",
                "ref_path": "",
                "method": "error",
                "orb_matches": 0,
                "error": str(e),
            })

    # Save CSV with useful info (but still simple)
    df = pd.DataFrame(flagged)
    df.to_csv(REPORT_CSV, index=False)

    # Write teammate-friendly HTML
    write_html_report_simple(flagged, title=f"PhotoSymbols PDF Scan — {pdf_path.name} (Flagged Only)")

    print("✅ Outputs created:")
    print(" -", REPORT_CSV)
    print(" -", REPORT_HTML)
    print(" - thumbnails in", THUMBS_DIR)
    return 0


# ---------------- main ----------------

def usage():
    print("Usage:")
    print('  python scanner.py index     "/path/to/PhotoSymbols"')
    print('  python scanner.py scan      "/path/to/Photos_To_Scan"')
    print('  python scanner.py scan_pdf  "/path/to/document.pdf"')


def main():
    if len(sys.argv) < 3:
        usage()
        return 1

    cmd = sys.argv[1].lower()
    target = Path(sys.argv[2]).expanduser()

    if not target.exists():
        print("❌ Not found:", target)
        return 1

    if cmd == "index":
        return build_reference_index(target)
    if cmd == "scan":
        return run_scan(target)
    if cmd == "scan_pdf":
        if target.suffix.lower() != ".pdf":
            print("❌ scan_pdf expects a .pdf file")
            return 1
        return run_scan_pdf(target)

    print("Unknown command:", cmd)
    usage()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())