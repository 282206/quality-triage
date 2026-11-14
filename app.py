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

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.common import RATER2_LABELS, RAW_DIR, REAL_LABELS, list_images, load_image
from src.dedup import DEFAULT_THRESHOLD, group_duplicates

st.set_page_config(page_title="Blur & Quality Triage", page_icon="📷", layout="wide")

BADGE = {"good": "#1a7f37", "borderline": "#9a6700", "bad": "#cf222e"}
TRASH = "_trash"
METHODS = ["Classical (texture-normalised)", "CNN"]
FILTERS = ["All", "Only bad", "Only duplicates"]
PAGE_SIZE = 60


def keep_widget_state(*keys: str) -> None:
    """Streamlit drops a keyed widget's value when the widget isn't rendered
    (e.g. on the other page). Re-assigning it every run keeps it alive."""
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


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
    return {k: (res[k] if k in ("sharp_pred", "exposure_pred", "phash") else float(res[k]))
            for k in ("sharp_pred", "exposure_pred", "sharp_score", "exposure_score", "quality", "phash")}


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=5000)
def thumbnail(path: str, mtime_ns: int, size: int = 320) -> Image.Image:
    im = _open_rgb(path)
    im.thumbnail((size, size))
    return im


@st.cache_data(show_spinner=False, max_entries=30)
def preview(path: str, mtime_ns: int, size: int = 1400) -> Image.Image:
    """Large image for the label page (small cache: these are big)."""
    im = _open_rgb(path)
    im.thumbnail((size, size))
    return im


def scan(folder: Path, method: str, dup_threshold: int) -> tuple[pd.DataFrame, list[str]]:
    """Analyse every image in `folder`. Returns (results worst-first, unreadable files)."""
    if method == "CNN":
        get_cnn()               # fail early (FileNotFoundError) if the model isn't trained
    else:
        get_classical()
    paths = list_images(folder)
    if not paths:
        return pd.DataFrame(), []
    bar = st.progress(0.0, text="Scanning...")
    rows, failed = [], []
    for i, p in enumerate(paths):
        try:
            mtime = p.stat().st_mtime_ns
            rows.append({"path": str(p), "filename": p.name, "mtime": mtime,
                         **analyse_one(str(p), mtime, method)})
        except Exception as e:   # one broken file must not kill the whole scan
            failed.append(f"{p.name}: {e}")
        bar.progress((i + 1) / len(paths), text=f"Scanning {i + 1}/{len(paths)}: {p.name}")
    bar.empty()
    if not rows:
        return pd.DataFrame(), failed
    df = pd.DataFrame(rows)
    df["bad"] = (df.sharp_pred == "bad") | (df.exposure_pred == "bad")
    df["dup_group"] = group_duplicates(df.phash.tolist(), dup_threshold)
    # In each duplicate group, keep the highest-quality image.
    df["keep"] = False
    grouped = df[df.dup_group >= 0]
    if len(grouped):
        df.loc[grouped.groupby("dup_group").quality.idxmax(), "keep"] = True
    return df.sort_values("quality").reset_index(drop=True), failed      # WORST first


# ============================================================================
# Trash: move (never delete), with a manifest so undo survives restarts
# ============================================================================
MANIFEST_COLS = ["batch", "original_path", "trash_path", "moved_at"]


def manifest_path(folder: Path) -> Path:
    return folder / TRASH / "manifest.csv"


def read_manifest(folder: Path) -> pd.DataFrame:
    mp = manifest_path(folder)
    if mp.exists():
        m = pd.read_csv(mp)
        return m[m.trash_path.map(lambda p: Path(p).exists())]   # drop entries already restored
    return pd.DataFrame(columns=MANIFEST_COLS)


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
    m = pd.concat([read_manifest(folder), pd.DataFrame(rows, columns=MANIFEST_COLS)], ignore_index=True)
    m.to_csv(manifest_path(folder), index=False)
    return len(rows)


def restore(folder: Path, entries: pd.DataFrame) -> tuple[int, list[str]]:
    """Move entries back. Returns (restored count, names skipped because the
    original location is occupied again)."""
    n, skipped = 0, []
    for r in entries.itertuples():
        src, dst = Path(r.trash_path), Path(r.original_path)
        if not src.exists():
            continue
        if dst.exists():
            skipped.append(dst.name)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n += 1
    read_manifest(folder).to_csv(manifest_path(folder), index=False)  # rewrites without restored rows
    return n, skipped


# ============================================================================
# Triage page - state helpers (all mutations go through callbacks, which run
# BEFORE the script, so every widget and metric on screen is consistent)
# ============================================================================
def sel_key(path: str) -> str:
    return f"sel::{path}"


def is_selected(path: str) -> bool:
    return bool(st.session_state.get(sel_key(path), False))


def set_selection(paths, value: bool) -> None:
    for p in paths:
        st.session_state[sel_key(p)] = value


def notify(msg: str, icon: str = "✅") -> None:
    st.session_state.setdefault("toasts", []).append((msg, icon))


def flush_toasts() -> None:
    for msg, icon in st.session_state.pop("toasts", []):
        st.toast(msg, icon=icon)


def request_scan() -> None:
    folder = Path(st.session_state.in_folder.strip()).expanduser()
    if not folder.is_dir():
        st.session_state.scan_error = f"Folder not found: `{folder}`"
        return
    st.session_state.scan_error = None
    st.session_state.scan_args = (str(folder.resolve()), st.session_state.in_method,
                                  int(st.session_state.in_thr))
    st.session_state.scan_dirty = True
    st.session_state.grid_page = 1
    st.session_state.pending_delete = None


def cb_confirm_move(folder: Path) -> None:
    pending = st.session_state.pending_delete or []
    moved = move_to_trash(folder, pending)
    for p in pending:
        st.session_state.pop(sel_key(p), None)
    st.session_state.pending_delete = None
    st.session_state.scan_dirty = True
    notify(f"Moved {moved} photo(s) to {TRASH}/  (undo from the sidebar)", "🗑")


def cb_restore(folder: Path, trash_paths: list[str] | None) -> None:
    m = read_manifest(folder)
    if trash_paths == ["__last__"]:
        m = m[m.batch == m.batch.max()]
    elif trash_paths is not None:
        m = m[m.trash_path.isin(trash_paths)]
    n, skipped = restore(folder, m)
    st.session_state.scan_dirty = True
    st.session_state.pop("restore_pick", None)
    notify(f"Restored {n} photo(s)", "↩")
    if skipped:
        notify(f"Not restored (a file with that name exists again): {', '.join(skipped)}", "⚠️")


# ============================================================================
# Triage page - rendering
# ============================================================================
def level(score: float) -> str:
    return "bad" if score < 1 / 3 else ("borderline" if score < 2 / 3 else "good")


def badge(text: str, lvl: str) -> str:
    return (f"<span style='background:{BADGE[lvl]};color:white;border-radius:4px;"
            f"padding:1px 6px;margin:0 4px 2px 0;font-size:0.78em;display:inline-block;"
            f"white-space:nowrap'>{text}</span>")


def card(row) -> None:
    with st.container(border=True):
        st.image(thumbnail(row.path, row.mtime), width="stretch")
        html = (badge(f"Q {row.quality:.2f}", level(row.quality))
                + badge(f"sharp: {row.sharp_pred}", row.sharp_pred)
                + badge(f"exp: {row.exposure_pred}", row.exposure_pred))
        if row.dup_group >= 0 and row.keep:
            html += badge("★ KEEP", "good")
        st.html(f"<div style='line-height:1.6'>{html}</div>")
        st.checkbox(row.filename, key=sel_key(row.path), persist_state="session",
                    help=f"sharpness score {row.sharp_score:.2f} · exposure score "
                         f"{row.exposure_score:.2f}\n\n{row.path}")


def grid(df: pd.DataFrame, n_cols: int) -> None:
    for start in range(0, len(df), n_cols):
        cols = st.columns(n_cols)
        for col, row in zip(cols, df.iloc[start:start + n_cols].itertuples()):
            with col:
                card(row)


def paged_grid(df: pd.DataFrame, n_cols: int) -> None:
    n_pages = max(1, -(-len(df) // PAGE_SIZE))
    if n_pages == 1:
        grid(df, n_cols)
        return
    page = min(st.session_state.get("grid_page", 1), n_pages)
    st.session_state.grid_page = page
    nav = st.columns([1, 2, 1])
    nav[0].button("◀ Previous", disabled=page <= 1, width="stretch",
                  on_click=lambda: st.session_state.update(grid_page=page - 1))
    nav[1].markdown(f"<div style='text-align:center;padding-top:6px'>Page <b>{page}</b> of "
                    f"{n_pages} · images {(page - 1) * PAGE_SIZE + 1}–"
                    f"{min(page * PAGE_SIZE, len(df))} of {len(df)}</div>", unsafe_allow_html=True)
    nav[2].button("Next ▶", disabled=page >= n_pages, width="stretch",
                  on_click=lambda: st.session_state.update(grid_page=page + 1))
    grid(df.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], n_cols)


def triage_page() -> None:
    st.title("Photo triage")
    st.caption("Scan a folder, review the worst photos and near-duplicates first, and move the ones "
               f"you don't want to `{TRASH}/` inside that folder. Nothing is ever deleted.")

    st.session_state.setdefault("in_folder", str(RAW_DIR))
    st.session_state.setdefault("in_method", METHODS[0])
    st.session_state.setdefault("in_thr", DEFAULT_THRESHOLD)
    st.session_state.setdefault("in_filter", "All")
    st.session_state.setdefault("in_cols", 5)

    with st.form("scan_form", border=False):
        c1, c2, c3, c4 = st.columns([5, 3, 2, 2], vertical_alignment="bottom")
        c1.text_input("Folder to scan", key="in_folder")
        c2.selectbox("Method", METHODS, key="in_method")
        c3.number_input("Duplicate threshold", 0, 32, key="in_thr",
                        help="Max pHash Hamming distance (of 64 bits) for two photos to count as "
                             "near-duplicates. Higher = more aggressive grouping.")
        c4.form_submit_button("Scan folder", type="primary", width="stretch", on_click=request_scan)

    if st.session_state.get("scan_error"):
        st.error(st.session_state.scan_error)
    if "scan_args" not in st.session_state:
        st.info("Choose a folder and press **Scan folder** (or Enter in the folder box).")
        return

    folder_s, method, thr = st.session_state.scan_args
    folder = Path(folder_s)
    if not folder.is_dir():
        st.error(f"Folder no longer exists: `{folder}`")
        return

    if st.session_state.get("scan_dirty", True) or "scan_df" not in st.session_state:
        try:
            df, failed = scan(folder, method, thr)
        except FileNotFoundError as e:
            st.error(f"Model not available: {e}")
            return
        st.session_state.scan_df, st.session_state.scan_failed = df, failed
        st.session_state.scan_time = time.strftime("%H:%M:%S")
        st.session_state.scan_dirty = False
        # forget selections of photos that are no longer in the folder
        live = set(df.path) if not df.empty else set()
        for k in [k for k in st.session_state if isinstance(k, str) and k.startswith("sel::")]:
            if k[5:] not in live:
                del st.session_state[k]
    df: pd.DataFrame = st.session_state.scan_df
    failed: list[str] = st.session_state.scan_failed

    flush_toasts()
    st.caption(f"Results for `{folder}` · {method} · dup. threshold {thr} · scanned at "
               f"{st.session_state.scan_time}")
    if failed:
        with st.expander(f"⚠️ {len(failed)} file(s) could not be read and were skipped"):
            st.code("\n".join(failed), language=None)

    if df.empty:
        st.warning("No readable images in this folder.")
        trash_controls(folder)
        return

    dup_mask = df.dup_group >= 0
    selected = [p for p in df.path if is_selected(p)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images", len(df))
    m2.metric("Flagged bad", int(df.bad.sum()), help="sharpness OR exposure predicted 'bad'")
    m3.metric("Duplicate groups", int(df.loc[dup_mask, "dup_group"].nunique()),
              help=f"{int(dup_mask.sum())} photos in groups; ★ KEEP marks the best of each group")
    m4.metric("Selected", len(selected))

    # --- toolbar ---------------------------------------------------------
    b1, b2, b3, flt_col, cols_col = st.columns([1.2, 1.4, 1.1, 2.6, 1.2], vertical_alignment="bottom")
    b1.button("Select all bad", width="stretch", disabled=not df.bad.any(),
              on_click=set_selection, args=(df.path[df.bad].tolist(), True))
    b2.button("Select dups (not KEEP)", width="stretch", disabled=not dup_mask.any(),
              on_click=set_selection, args=(df.path[dup_mask & ~df.keep].tolist(), True))
    b3.button("Clear selection", width="stretch", disabled=not selected,
              on_click=set_selection, args=(df.path.tolist(), False))
    flt = flt_col.segmented_control("Show", FILTERS, key="in_filter") or "All"
    n_cols = cols_col.selectbox("Per row", [3, 4, 5, 6, 8], key="in_cols")
    if st.session_state.get("last_filter") != flt:       # new filter -> back to page 1
        st.session_state.last_filter, st.session_state.grid_page = flt, 1

    delete_controls(folder, selected)

    # --- results ---------------------------------------------------------
    if flt != "Only bad":
        dups = df[dup_mask]
        if len(dups):
            st.subheader(f"Duplicate groups ({dups.dup_group.nunique()})")
            st.caption("Worst group first. Inside each group the ★ KEEP photo (highest quality) is shown first.")
            order = dups.groupby("dup_group").quality.min().sort_values().index
            for g in order:
                members = dups[dups.dup_group == g].sort_values("quality", ascending=False)
                keep = members[members.keep].filename.iloc[0]
                n_sel = sum(is_selected(p) for p in members.path)
                label = f"{len(members)} similar photos · keep **{keep}**"
                if n_sel:
                    label += f" · {n_sel} selected"
                with st.expander(label):
                    grid(members, n_cols)
        elif flt == "Only duplicates":
            st.info("No near-duplicates found at this threshold.")

    if flt != "Only duplicates":
        view = df[df.bad] if flt == "Only bad" else df[~dup_mask]
        st.subheader("Bad images, worst first" if flt == "Only bad" else "Other images, worst first")
        if view.empty:
            st.success("Nothing flagged bad. 🎉" if flt == "Only bad" else "All images are in duplicate groups.")
        else:
            paged_grid(view, n_cols)

    trash_controls(folder)


def delete_controls(folder: Path, selected: list[str]) -> None:
    pending = st.session_state.get("pending_delete")
    if pending:
        with st.container(border=True):
            st.warning(f"Move **{len(pending)}** photo(s) to `{folder / TRASH}`? "
                       "They are not deleted; you can restore them from the sidebar.")
            y, n, _ = st.columns([1.3, 1, 5])
            y.button("Yes, move them", type="primary", width="stretch",
                     on_click=cb_confirm_move, args=(folder,))
            n.button("Cancel", width="stretch",
                     on_click=lambda: st.session_state.update(pending_delete=None))
    else:
        st.button(f"🗑 Move {len(selected)} selected to {TRASH}/", disabled=not selected,
                  type="primary" if selected else "secondary",
                  on_click=lambda: st.session_state.update(pending_delete=list(selected)))


def trash_controls(folder: Path) -> None:
    m = read_manifest(folder)
    with st.sidebar.expander(f"🗑 Trash ({len(m)} photo{'s' if len(m) != 1 else ''})", expanded=len(m) > 0):
        if m.empty:
            st.caption("Nothing in the trash.")
            return
        last = m[m.batch == m.batch.max()]
        st.button(f"↩ Undo last move ({len(last)})", width="stretch",
                  on_click=cb_restore, args=(folder, ["__last__"]))
        st.button("Restore all", width="stretch", on_click=cb_restore, args=(folder, None))
        pick = st.multiselect("Restore specific photos", m.trash_path.tolist(), key="restore_pick",
                              format_func=lambda p: Path(p).name)
        st.button("Restore selected", width="stretch", disabled=not pick,
                  on_click=cb_restore, args=(folder, pick))
        st.caption(f"Files live in `{folder / TRASH}`; empty it yourself when you are sure.")


# ============================================================================
# Label page
# ============================================================================
KEYS = {"sharpness": ["1", "2", "3"], "exposure": ["Q", "W", "E"]}
LABEL_CLASSES = ["good", "borderline", "bad"]


def load_labels(csv: Path) -> pd.DataFrame:
    if csv.exists():
        df = pd.read_csv(csv, dtype=str)
        return df.drop_duplicates("filename", keep="last").reset_index(drop=True)
    return pd.DataFrame(columns=["filename", "sharpness", "exposure", "rater", "timestamp"])


def save_label(csv: Path, filename: str, field: str, value: str, rater: str) -> None:
    """Upsert one field for one image and write the CSV immediately."""
    df = load_labels(csv)
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for c in (field, "rater", "timestamp"):
        if c not in df.columns:
            df[c] = pd.Series(dtype=str)
    if (df.filename == filename).any():
        idx = df.index[df.filename == filename][0]
        df.loc[idx, [field, "rater", "timestamp"]] = [value, rater, now]
    else:
        df = pd.concat([df, pd.DataFrame([{"filename": filename, field: value, "rater": rater,
                                           "timestamp": now}])], ignore_index=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv)                     # atomic: a crash never leaves a half-written CSV


def keyboard_shortcuts() -> None:
    """Tiny JS bridge: key presses click the matching Streamlit button.
    1/2/3 = sharpness good/borderline/bad, Q/W/E = exposure, ->/N next, <-/B back.
    The handler is replaced (not stacked) on every load and removed when the
    iframe goes away, so it keeps working after switching pages."""
    st.iframe("""
    <script>
    const doc = window.parent.document;
    const map = {'1':'[1]','2':'[2]','3':'[3]','q':'[Q]','w':'[W]','e':'[E]',
                 'arrowright':'Next','n':'Next','arrowleft':'Back','b':'Back'};
    const handler = (ev) => {
      if (ev.metaKey || ev.ctrlKey || ev.altKey || ev.repeat) return;
      const t = ev.target;
      if (t && (['INPUT','TEXTAREA','SELECT'].includes(t.tagName) || t.isContentEditable)) return;
      const tag = map[ev.key.toLowerCase()];
      if (!tag) return;
      const btn = Array.from(doc.querySelectorAll('button'))
                       .find(b => !b.disabled && b.innerText.includes(tag));
      if (btn) { btn.click(); ev.preventDefault(); }
    };
    if (doc._triageKeyHandler) doc.removeEventListener('keydown', doc._triageKeyHandler);
    doc._triageKeyHandler = handler;
    doc.addEventListener('keydown', handler);
    window.addEventListener('pagehide', () => {
      if (doc._triageKeyHandler === handler) {
        doc.removeEventListener('keydown', handler);
        doc._triageKeyHandler = null;
      }
    });
    </script>""", height=1)


def label_page() -> None:
    st.title("Label photos")
    st.session_state.setdefault("label_skip_done", True)
    target = st.sidebar.radio("Save labels to", ["Rater 1 → real_labels.csv", "Rater 2 → rater2_labels.csv"],
                              key="label_target")
    csv = REAL_LABELS if target.startswith("Rater 1") else RATER2_LABELS
    rater = st.sidebar.text_input("Rater name", key="rater_name", placeholder="e.g. alice").strip()
    hide_done = st.sidebar.checkbox("Skip already-labelled photos on Next", key="label_skip_done")
    st.sidebar.markdown("**Keyboard**  \n`1` `2` `3` sharpness good / borderline / bad  \n"
                        "`Q` `W` `E` exposure good / borderline / bad  \n`→` / `N` next · `←` / `B` back")

    images = list_images(RAW_DIR)
    if not images:
        st.info(f"Put photos in `{RAW_DIR}` first.")
        return
    labels = load_labels(csv).set_index("filename")

    def value(f: str, field: str) -> str | None:
        if f not in labels.index or field not in labels.columns:
            return None
        v = labels.at[f, field]
        return v if isinstance(v, str) and v in LABEL_CLASSES else None

    def done(f: str) -> bool:
        return value(f, "sharpness") is not None and value(f, "exposure") is not None

    n_done = sum(done(p.name) for p in images)
    key = f"label_idx::{csv.name}"
    if key not in st.session_state:   # start at the first unlabelled photo
        st.session_state[key] = next((i for i, p in enumerate(images) if not done(p.name)), 0)
    idx = max(0, min(st.session_state[key], len(images) - 1))
    st.session_state[key] = idx
    img = images[idx]

    def go(step: int) -> None:
        i = idx + step
        if hide_done and step > 0:
            while i < len(images) and done(images[i].name):
                i += 1
            if i >= len(images):                  # nothing unlabelled ahead: wrap to the first one
                i = next((j for j, p in enumerate(images) if not done(p.name)), idx + step)
        st.session_state[key] = max(0, min(len(images) - 1, i))

    def on_label(field: str, cls: str) -> None:
        save_label(csv, img.name, field, cls, rater)
        other = "exposure" if field == "sharpness" else "sharpness"
        # auto-advance only when this click COMPLETES the photo (revising doesn't jump away)
        if not done(img.name) and value(img.name, other) is not None:
            labels.loc[img.name, field] = cls      # so go() sees this one as done
            go(+1)

    st.progress(n_done / len(images), text=f"{n_done} / {len(images)} labelled")
    if n_done == len(images):
        st.success(f"All {len(images)} photos are labelled in `{csv.name}`. 🎉 You can still revise any of them.")
    if not rater:
        st.warning("Enter your **rater name** in the sidebar to enable the label buttons.")

    left, right = st.columns([3, 1])
    with left:
        im = preview(str(img), img.stat().st_mtime_ns)
        st.markdown(f"**{img.name}** · photo {idx + 1} of {len(images)}"
                    + ("  ✅ labelled" if done(img.name) else ""))
        # keep portrait photos within the screen height
        st.image(im, width="stretch" if im.width >= im.height else int(720 * im.width / im.height))
    with right:
        for field in ("sharpness", "exposure"):
            cur = value(img.name, field)
            st.markdown(f"**{field.capitalize()}**" + (f" · saved: *{cur}*" if cur else " · *not set*"))
            for cls, k in zip(LABEL_CLASSES, KEYS[field]):
                st.button(f"{cls} [{k}]", key=f"{field}_{cls}", type="primary" if cur == cls else "secondary",
                          width="stretch", disabled=not rater, on_click=on_label, args=(field, cls))
        st.divider()
        b, n = st.columns(2)
        b.button("◀ Back", width="stretch", disabled=idx == 0, on_click=go, args=(-1,))
        n.button("Next ▶", width="stretch", on_click=go, args=(+1,))
        st.selectbox("Jump to photo", range(len(images)), index=idx, key=f"jump::{csv.name}::{idx}",
                     format_func=lambda i: f"{i + 1}. {images[i].name}" + ("  ✓" if done(images[i].name) else ""),
                     on_change=lambda: st.session_state.update(
                         {key: st.session_state[f"jump::{csv.name}::{idx}"]}))
        st.caption(f"Every click is saved to `{csv.name}` immediately.")
    keyboard_shortcuts()


# ============================================================================
# keep both pages' inputs alive while the other page is shown
keep_widget_state("in_folder", "in_method", "in_thr", "in_filter", "in_cols",
                  "rater_name", "label_target", "label_skip_done")
page = st.sidebar.radio("Page", ["Triage", "Label"], key="page", horizontal=True)
st.sidebar.divider()
if page == "Triage":
    triage_page()
else:
    label_page()
