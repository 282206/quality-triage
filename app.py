"""Phase 8 - Streamlit app: triage a photo folder, and label photos.

Run:  streamlit run app.py

Triage page : scan a folder, rank worst-first, group near-duplicates, select
              images and MOVE them to <folder>/_trash/ (never deleted), with
              confirmation and undo/restore.
Label page  : label data/raw photos one at a time (sharpness + exposure);
              every click is saved to CSV immediately.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from src.common import CLASSES, RATER2_LABELS, RAW_DIR, REAL_LABELS, list_images, load_image
from src.dedup import DEFAULT_THRESHOLD, group_duplicates

st.set_page_config(page_title="Blur & Quality Triage", layout="wide")

BADGE = {"good": "#1a7f37", "borderline": "#b08800", "bad": "#cf222e"}
TRASH = "_trash"


# ============================================================================
# Analysis (cached per file, so re-scans only process new/changed photos)
# ============================================================================
@st.cache_resource
def get_cnn():
    from src.cnn import load_cnn
    return load_cnn()


@st.cache_resource
def get_classical():
    from src.baseline import load_classical
    return load_classical("norm")


@st.cache_data(show_spinner=False, max_entries=20000)
def analyse_one(path: str, mtime_ns: int, method: str) -> dict:
    """Predictions + pHash for one image. mtime_ns is part of the cache key,
    so an edited file is re-analysed."""
    from src.features import phash_hex

    if method == "CNN":
        from src.cnn import predict_path
        res = predict_path(get_cnn(), path)
    else:
        from src.baseline import predict_path
        res = predict_path(path, get_classical())
    res["phash"] = phash_hex(load_image(path))
    return {k: res[k] for k in ("sharp_pred", "exposure_pred", "sharp_score",
                                "exposure_score", "quality", "phash")}


@st.cache_data(show_spinner=False, max_entries=20000)
def thumbnail(path: str, mtime_ns: int, size: int = 320) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((size, size))
    return im


def scan(folder: Path, method: str, dup_threshold: int) -> pd.DataFrame:
    paths = list_images(folder)
    if not paths:
        return pd.DataFrame()
    bar = st.progress(0.0, text="Scanning...")
    rows = []
    for i, p in enumerate(paths):
        rows.append({"path": str(p), "filename": p.name, "mtime": p.stat().st_mtime_ns,
                     **analyse_one(str(p), p.stat().st_mtime_ns, method)})
        bar.progress((i + 1) / len(paths), text=f"Scanning {i + 1}/{len(paths)}: {p.name}")
    bar.empty()
    df = pd.DataFrame(rows)
    df["bad"] = (df.sharp_pred == "bad") | (df.exposure_pred == "bad")
    df["dup_group"] = group_duplicates(df.phash.tolist(), dup_threshold)
    # In each duplicate group, keep the highest-quality image.
    df["keep"] = False
    grouped = df[df.dup_group >= 0]
    if len(grouped):
        df.loc[grouped.groupby("dup_group").quality.idxmax(), "keep"] = True
    return df.sort_values("quality").reset_index(drop=True)      # WORST first


# ============================================================================
# Trash: move (never delete), with a manifest so undo survives restarts
# ============================================================================
def manifest_path(folder: Path) -> Path:
    return folder / TRASH / "manifest.csv"


def read_manifest(folder: Path) -> pd.DataFrame:
    mp = manifest_path(folder)
    if mp.exists():
        m = pd.read_csv(mp)
        return m[m.trash_path.map(lambda p: Path(p).exists())]   # drop entries already restored
    return pd.DataFrame(columns=["batch", "original_path", "trash_path", "moved_at"])


def move_to_trash(folder: Path, paths: list[str]) -> int:
    trash = folder / TRASH
    trash.mkdir(exist_ok=True)
    batch = int(time.time() * 1000)
    rows = []
    for p in map(Path, paths):
        if not p.exists():
            continue
        dest = trash / p.name
        k = 1
        while dest.exists():                       # never overwrite something in the trash
            dest = trash / f"{p.stem}__{k}{p.suffix}"
            k += 1
        shutil.move(str(p), str(dest))
        rows.append({"batch": batch, "original_path": str(p), "trash_path": str(dest),
                     "moved_at": pd.Timestamp.now().isoformat(timespec="seconds")})
    m = pd.concat([read_manifest(folder), pd.DataFrame(rows)], ignore_index=True)
    m.to_csv(manifest_path(folder), index=False)
    return len(rows)


def restore(folder: Path, entries: pd.DataFrame) -> int:
    n = 0
    for r in entries.itertuples():
        src, dst = Path(r.trash_path), Path(r.original_path)
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            n += 1
    read_manifest(folder).to_csv(manifest_path(folder), index=False)  # rewrites without restored rows
    return n


# ============================================================================
# Triage page
# ============================================================================
def badge(text: str, level: str) -> str:
    return (f"<span style='background:{BADGE[level]};color:white;border-radius:4px;"
            f"padding:1px 6px;margin-right:4px;font-size:0.8em'>{text}</span>")


def card(row) -> None:
    st.image(thumbnail(row.path, row.mtime), width="stretch")
    q_level = "bad" if row.quality < 1 / 3 else ("borderline" if row.quality < 2 / 3 else "good")
    html = (badge(f"Q {row.quality:.2f}", q_level) + badge(f"sharp: {row.sharp_pred}", row.sharp_pred)
            + badge(f"exp: {row.exposure_pred}", row.exposure_pred))
    if row.dup_group >= 0 and row.keep:
        html += badge("KEEP", "good")
    st.markdown(html, unsafe_allow_html=True)
    st.checkbox(row.filename, key=f"sel::{row.path}")


def grid(df: pd.DataFrame, n_cols: int = 5) -> None:
    for start in range(0, len(df), n_cols):
        cols = st.columns(n_cols)
        for col, row in zip(cols, df.iloc[start:start + n_cols].itertuples()):
            with col:
                card(row)


def triage_page() -> None:
    st.title("Photo triage")
    c1, c2, c3 = st.columns([4, 2, 1])
    folder = Path(c1.text_input("Folder to scan", str(RAW_DIR))).expanduser()
    method = c2.selectbox("Method", ["Classical (texture-normalised)", "CNN"])
    dup_thr = c3.number_input("Dup. threshold", 0, 32, DEFAULT_THRESHOLD,
                              help="max pHash Hamming distance (of 64 bits) to call two photos duplicates")
    if st.button("Scan folder", type="primary"):
        st.session_state.scan_args = (str(folder), method, int(dup_thr))
    if "scan_args" not in st.session_state:
        st.info("Choose a folder and press **Scan folder**.")
        return
    folder = Path(st.session_state.scan_args[0])
    if not folder.is_dir():
        st.error(f"Folder not found: {folder}")
        return
    try:
        df = scan(folder, st.session_state.scan_args[1], st.session_state.scan_args[2])
    except FileNotFoundError as e:
        st.error(str(e))
        return
    if df.empty:
        st.warning("No images in this folder.")
    else:
        n_groups = df.loc[df.dup_group >= 0, "dup_group"].nunique()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images", len(df))
        m2.metric("Flagged bad", int(df.bad.sum()))
        m3.metric("Duplicate groups", n_groups)
        m4.metric("Selected", sum(st.session_state.get(f"sel::{p}", False) for p in df.path))

        # --- quick selection helpers (set checkbox state BEFORE they render)
        b1, b2, b3, show = st.columns([1, 1, 1, 3])
        if b1.button("Select all bad"):
            for p in df.path[df.bad]:
                st.session_state[f"sel::{p}"] = True
        if b2.button("Select dups (not KEEP)"):
            for p in df.path[(df.dup_group >= 0) & ~df.keep]:
                st.session_state[f"sel::{p}"] = True
        if b3.button("Clear selection"):
            for p in df.path:
                st.session_state[f"sel::{p}"] = False
        flt = show.radio("Show", ["All", "Only bad", "Only duplicates"], horizontal=True)

        selected = [p for p in df.path if st.session_state.get(f"sel::{p}", False)]
        delete_controls(folder, selected)

        view = df[df.bad] if flt == "Only bad" else df
        dups = view[view.dup_group >= 0]
        if flt != "Only bad" and len(dups):
            st.subheader("Duplicate groups")
            order = dups.groupby("dup_group").quality.min().sort_values().index   # worst group first
            for g in order:
                members = dups[dups.dup_group == g]
                keep = df[(df.dup_group == g) & df.keep].filename.iloc[0]
                with st.expander(f"Group {g}: {len(df[df.dup_group == g])} similar photos  (keep: {keep})"):
                    grid(members)
        if flt != "Only duplicates":
            singles = view[view.dup_group < 0] if flt == "All" else view
            st.subheader("Images, worst first" if flt == "All" else "Bad images, worst first")
            grid(singles)
    trash_controls(folder)


def delete_controls(folder: Path, selected: list[str]) -> None:
    if st.session_state.get("pending_delete"):
        pending = st.session_state.pending_delete
        st.warning(f"Move **{len(pending)}** photo(s) to `{folder / TRASH}`? "
                   "(They are not deleted; you can restore them below.)")
        y, n, _ = st.columns([1, 1, 6])
        if y.button("Yes, move them", type="primary"):
            moved = move_to_trash(folder, pending)
            for p in pending:
                st.session_state.pop(f"sel::{p}", None)
            st.session_state.pending_delete = None
            st.toast(f"Moved {moved} photo(s) to {TRASH}/")
            st.rerun()
        if n.button("Cancel"):
            st.session_state.pending_delete = None
            st.rerun()
    elif st.button(f"🗑 Move {len(selected)} selected to {TRASH}/", disabled=not selected):
        st.session_state.pending_delete = selected
        st.rerun()


def trash_controls(folder: Path) -> None:
    m = read_manifest(folder)
    with st.sidebar.expander(f"Trash ({len(m)} photos)", expanded=len(m) > 0):
        if m.empty:
            st.caption("Nothing in the trash.")
            return
        if st.button("↩ Undo last move"):
            n = restore(folder, m[m.batch == m.batch.max()])
            st.toast(f"Restored {n} photo(s)")
            st.rerun()
        if st.button("Restore all"):
            n = restore(folder, m)
            st.toast(f"Restored {n} photo(s)")
            st.rerun()
        pick = st.multiselect("Restore specific photos", m.trash_path.tolist(),
                              format_func=lambda p: Path(p).name)
        if pick and st.button("Restore selected"):
            restore(folder, m[m.trash_path.isin(pick)])
            st.rerun()
        st.caption(f"Files live in `{folder / TRASH}`; empty it yourself when you are sure.")


# ============================================================================
# Label page
# ============================================================================
KEYS = {"sharpness": ["1", "2", "3"], "exposure": ["Q", "W", "E"]}


def load_labels(csv: Path) -> pd.DataFrame:
    if csv.exists():
        return pd.read_csv(csv, dtype=str)
    return pd.DataFrame(columns=["filename", "sharpness", "exposure", "rater", "timestamp"])


def save_label(csv: Path, filename: str, field: str, value: str, rater: str) -> None:
    """Upsert one field for one image and write the CSV immediately."""
    df = load_labels(csv)
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    if (df.filename == filename).any():
        idx = df.index[df.filename == filename][0]
        df.loc[idx, [field, "rater", "timestamp"]] = [value, rater, now]
    else:
        df = pd.concat([df, pd.DataFrame([{"filename": filename, field: value, "rater": rater,
                                           "timestamp": now}])], ignore_index=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)


def keyboard_shortcuts() -> None:
    """Tiny JS bridge: key presses click the matching Streamlit button.
    1/2/3 = sharpness good/borderline/bad, Q/W/E = exposure, ->/N next, <-/B back."""
    components.html("""
    <script>
    const doc = window.parent.document;
    if (!doc._triageKeys) {
      doc._triageKeys = true;
      const map = {'1':'[1]','2':'[2]','3':'[3]','q':'[Q]','w':'[W]','e':'[E]',
                   'arrowright':'Next','n':'Next','arrowleft':'Back','b':'Back'};
      doc.addEventListener('keydown', (ev) => {
        if (['INPUT','TEXTAREA'].includes(ev.target.tagName)) return;
        const tag = map[ev.key.toLowerCase()];
        if (!tag) return;
        const btn = Array.from(doc.querySelectorAll('button'))
                         .find(b => b.innerText.includes(tag));
        if (btn) { btn.click(); ev.preventDefault(); }
      });
    }
    </script>""", height=0)


def label_page() -> None:
    st.title("Label photos")
    target = st.sidebar.radio("Save labels to", ["Rater 1 → real_labels.csv", "Rater 2 → rater2_labels.csv"])
    csv = REAL_LABELS if target.startswith("Rater 1") else RATER2_LABELS
    rater = st.sidebar.text_input("Rater name", st.session_state.get("rater_name", ""))
    st.session_state.rater_name = rater
    hide_done = st.sidebar.checkbox("Skip already-labelled photos on Next", value=True)
    st.sidebar.caption("Keys: **1/2/3** sharpness good/borderline/bad · **Q/W/E** exposure · "
                       "**→ / N** next · **← / B** back")

    images = list_images(RAW_DIR)
    if not images:
        st.info(f"Put photos in `{RAW_DIR}` first.")
        return
    labels = load_labels(csv).set_index("filename")
    done = lambda f: f in labels.index and all(isinstance(labels.loc[f, c], str) for c in ("sharpness", "exposure"))
    n_done = sum(done(p.name) for p in images)

    key = f"label_idx::{csv.name}"
    if key not in st.session_state:   # start at the first unlabelled photo
        st.session_state[key] = next((i for i, p in enumerate(images) if not done(p.name)), 0)
    idx = min(st.session_state[key], len(images) - 1)
    img = images[idx]

    st.progress(n_done / len(images), text=f"{n_done} / {len(images)} labelled  ·  photo {idx + 1}: {img.name}")
    if not rater.strip():
        st.warning("Enter your rater name in the sidebar before labelling.")

    def go(step: int) -> None:
        i = idx + step
        if hide_done and step > 0:
            while i < len(images) - 1 and done(images[i].name):
                i += 1
        st.session_state[key] = max(0, min(len(images) - 1, i))

    left, right = st.columns([3, 1])
    with left:
        st.image(thumbnail(str(img), img.stat().st_mtime_ns, size=1400), width="stretch")
    with right:
        current = labels.loc[img.name] if img.name in labels.index else {}
        for field in ("sharpness", "exposure"):
            st.markdown(f"**{field.capitalize()}**")
            cur = current.get(field) if hasattr(current, "get") else None
            for cls, k in zip(["good", "borderline", "bad"], KEYS[field]):
                kind = "primary" if cur == cls else "secondary"
                if st.button(f"{cls} [{k}]", key=f"{field}_{cls}", type=kind,
                             width="stretch", disabled=not rater.strip()):
                    save_label(csv, img.name, field, cls, rater.strip())
                    other = "exposure" if field == "sharpness" else "sharpness"
                    other_val = current.get(other) if hasattr(current, "get") else None
                    if isinstance(other_val, str):     # both labels now set -> next photo
                        go(+1)
                    st.rerun()
        b, n = st.columns(2)
        if b.button("◀ Back", width="stretch"):
            go(-1)
            st.rerun()
        if n.button("Next ▶", width="stretch"):
            go(+1)
            st.rerun()
        st.caption(f"Saving to `{csv.name}` after every click.")
    keyboard_shortcuts()


# ============================================================================
page = st.sidebar.radio("Page", ["Triage", "Label"])
if page == "Triage":
    triage_page()
else:
    label_page()
