import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set

# --------------------------- Optional heavy libs ---------------------------
try:
    import pdfplumber
    _PDF_OK = True
except Exception:
    pdfplumber = None
    _PDF_OK = False

try:
    import docx as _docx
    _DOCX_OK = True
except Exception:
    _docx = None
    _DOCX_OK = False

try:
    import rdflib
    _RDFLIB_OK = True
except Exception:
    rdflib = None
    _RDFLIB_OK = False

try:
    import networkx as nx
    _NX_OK = True
except Exception:
    nx = None
    _NX_OK = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    _SK_OK = True
except Exception:
    cosine_similarity = None
    _SK_OK = False

try:
    import plotly.express as px
    _PX_OK = True
    px.defaults.template = "plotly_white"
except Exception:
    px = None
    _PX_OK = False

try:
    from transformers import pipeline
    _HF_OK = True
except Exception:
    pipeline = None
    _HF_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception:
    SentenceTransformer = None
    _SBERT_OK = False

# --------------------------- App config & style ---------------------------
st.set_page_config(
    page_title="Policy Alignment Dasboard - Sultanate of Oman",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.2rem;box-shadow:0 1px 2px rgba(0,0,0,.05)}
      .stProgress > div > div > div > div{background-image:linear-gradient(90deg,#2563eb,#7c3aed)}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Policy Alignment Dashboard - Sultanate of Oman")
st.caption("Notebook-faithful pipeline: ZSC only for Vision 2040; SBERT + UNBIS mapping; Wu–Palmer BMA. Aggressive cleaning to align results.")

# --------------------------- Dependency tips ---------------------------
missing = []
if not _PDF_OK: missing.append("pdfplumber")
if not _DOCX_OK: missing.append("python-docx")
if not _RDFLIB_OK: missing.append("rdflib")
if not _NX_OK: missing.append("networkx")
if not _SK_OK: missing.append("scikit-learn")
if not _PX_OK: missing.append("plotly")
if not _HF_OK: missing.append("transformers")
if not _SBERT_OK: missing.append("sentence-transformers")
if missing:
    with st.expander("Install instructions (missing deps)"):
        st.code("pip install streamlit pdfplumber python-docx rdflib networkx scikit-learn plotly transformers sentence-transformers", language="bash")
        for m in missing: st.write("•", m)

# --------------------------- Status shim ---------------------------
def status_block(title: str):
    try:
        return st.status(title, expanded=True)
    except Exception:
        c = st.container()
        c.markdown(f'<div class="card"><b>{title}</b></div>', unsafe_allow_html=True)
        class Dummy:
            def update(self, label: str = None, state: str = None):
                if label: c.write(label)
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): pass
        return Dummy()

# --------------------------- Constants & cleaning ---------------------------
AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
SENT_SPLIT = re.compile(r"(?<=[\.!\؟\?\!\؛\;])\s+")

# NOTE: exact notebook order + naming
AXES_EN = [
    "Man and society",
    "Economy and Development",
    "Governance and Institutional Performance",
    "Environment and Sustainability",
]
AX_TO_THEME = {
    "Man and society": "soc",
    "Economy and Development": "econ",
    "Governance and Institutional Performance": "gov",
    "Environment and Sustainability": "env",
}

PLAN_NAME_PATTERNS = {
    "Vision 2040": r"vision\s*20?40|oman\s*vision",
    "Economic Stimulus Plan": r"economic\s*stimulus|stimulus\s*plan",
    "Medium Term Fiscal Plan": r"medium\s*term\s*fiscal|mtfp",
    "National Digital Economy Program (summary)": r"digital\s*economy.*summary|ndep|digital\s*economy\s*program",
    "Manufacturing Strategy": r"manufacturing\s*strategy(?!.*2040)|manufacturing\s*strategy\s*$",
    "Strategy for an Orderly Transition to Net Zero": r"net\s*zero|orderly\s*transition",
    "Fisheries & Aquaculture Strategy": r"fisheries\s*&?\s*aquaculture(?!.*2040)|aquaculture\s*strategy",
    "National Strategy for Education": r"education\s*strategy|strategy\s*for\s*education",
}

# sector lists (NDEP removed from visuals)
ECO_LIST = ["Economic Stimulus Plan", "Medium Term Fiscal Plan", "Manufacturing Strategy"]
ENV_LIST = ["Strategy for an Orderly Transition to Net Zero", "Fisheries & Aquaculture Strategy"]
GOV_LIST = ["National Strategy for Education", "Medium Term Fiscal Plan", "Economic Stimulus Plan"]
SOC_LIST = ["National Strategy for Education"]

AR_EN_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~،؛؟“”‘’"""
AR_STOP = set("من في على و أو ثم بل مع عن إلى الى حتى لدى عند هذا هذه ذلك تلك هناك هنا كان تكون تكونوا كانت كانوا كون إن أن إنّ أنّ إذا اذا ما لا لم لن ألا إلا إلاّ غير دون بين كما حيث لكن لأن بأن الى من قبل بعد ضد حسب منذ خلال مثلا مثل كثير جدا جداً جداَ أيضاً ايضا فقط قد كل كافة كافةً كذلك لذا لكنّ لكي لكيما كي حين سواء سواءً اي ايضاً أي ألا وهؤلاء هؤلاء أولئك الى إنّما نعم لا نعمَ كلا كلاّ إذ إذن إذًا سوى سوىً كأن كأنّ".split())
EN_STOP = set("a an and are as at be by for from has have if in into is it its of on or that the their there these this to was were will with without within while about across after again against among around because before being both cannot did do does doing down during each few further here how i me more most my myself no nor not now once only other our out over own same she should so some such than then there they through too under until up very we what when where which who whom why you your yours yourself yourselves".split())

def normalize_arabic(s: str) -> str:
    if not s: return s
    s = AR_DIACRITICS.sub("", s)
    s = s.replace("ـ", "")
    s = re.sub("[إأٱآا]", "ا", s)
    s = s.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    return s

def strip_urls_emails_nums(text: str) -> str:
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d+([.,:/-]\d+)*\b", " ", text)
    return text

def remove_bullets_headers_footers(text: str) -> str:
    text = re.sub(r"^\s*(figure|table|appendix|annex|reference|contents)\b.*$", " ", text, flags=re.I|re.M)
    text = re.sub(r"^\s*page\s*\d+\s*$", " ", text, flags=re.I|re.M)
    text = re.sub(r"^\s*[-•●▪▶*]+\s*", " ", text, flags=re.M)
    return text

def rm_punct(text: str) -> str:
    return re.sub(f"[{re.escape(AR_EN_PUNCT)}]", " ", text)

def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def tokenize_simple(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text.lower()) if t]

def drop_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if (t not in EN_STOP and t not in AR_STOP and len(t) > 2)]

def aggressive_clean(text: str) -> str:
    if text is None: return ""
    t = strip_urls_emails_nums(text)
    t = remove_bullets_headers_footers(t)
    t = normalize_arabic(t)
    t = rm_punct(t)
    t = collapse(t)
    toks = drop_stopwords(tokenize_simple(t))
    return " ".join(toks)

def basic_clean(text: str) -> str:
    if text is None: return ""
    t = re.sub(r"\s+", " ", text).replace("\u200f","").replace("\u200e","")
    t = normalize_arabic(t)
    return t.strip()

def split_sentences(text: str) -> List[str]:
    parts = SENT_SPLIT.split(text)
    return [s.strip() for s in parts if s and len(s.strip()) > 3]

# --------------------------- Readers ---------------------------
def read_pdf_bytes(b: bytes) -> str:
    if not _PDF_OK: return ""
    out = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(out)

def read_docx_bytes(b: bytes) -> str:
    if not _DOCX_OK: return ""
    doc = _docx.Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt_bytes(b: bytes) -> str:
    for enc in ("utf-8", "cp1256", "latin-1"):
        try: return b.decode(enc)
        except Exception: continue
    return ""

@st.cache_data(show_spinner=True)
def read_plans_zip(zip_bytes: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            name = info.filename
            suffix = Path(name).suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt"}: continue
            b = z.read(info)
            try:
                if suffix == ".pdf": txt = read_pdf_bytes(b)
                elif suffix == ".docx": txt = read_docx_bytes(b)
                else: txt = read_txt_bytes(b)
            except Exception:
                txt = ""
            txt = basic_clean(txt)
            if txt: out[name] = txt
    return out

# --------------------------- Models ---------------------------
@st.cache_resource(show_spinner=True)
def load_zsc(model_name: str):
    if not _HF_OK: raise RuntimeError("transformers not installed.")
    return pipeline("zero-shot-classification", model=model_name, device=-1)

@st.cache_resource(show_spinner=True)
def load_sbert(model_name: str):
    if not _SBERT_OK: raise RuntimeError("sentence-transformers not installed.")
    return SentenceTransformer(model_name)

# --------------------------- UNBIS parsing & embeddings ---------------------------
@st.cache_data(show_spinner=True)
def parse_unbis_ttl(ttl_bytes: bytes) -> Tuple[Any, Dict[str, Dict[str, List[str]]]]:
    if not _RDFLIB_OK: raise RuntimeError("rdflib is required.")
    if not _NX_OK: raise RuntimeError("networkx is required.")
    g = rdflib.Graph()
    g.parse(data=ttl_bytes.decode("utf-8", errors="ignore"), format="turtle")
    SKOS = rdflib.namespace.SKOS

    G = nx.DiGraph()
    concept_labels: Dict[str, Dict[str, List[str]]] = {}

    for s in g.subjects(rdflib.RDF.type, SKOS.Concept):
        sid = str(s)
        G.add_node(sid)
        pref = [str(o) for o in g.objects(s, SKOS.prefLabel)]
        alt  = [str(o) for o in g.objects(s, SKOS.altLabel)]
        concept_labels[sid] = {"pref": pref, "alt": alt}

    for s, o in g.subject_objects(SKOS.broader):
        # s broader o => edge o -> s (parent -> child)
        G.add_edge(str(o), str(s))
    return G, concept_labels

@st.cache_resource(show_spinner=True)
def embed_concepts(concept_labels: Dict[str, Dict[str, List[str]]], sbert_model_name: str):
    model = load_sbert(sbert_model_name)
    nodes = list(concept_labels.keys())
    texts, idx_map = [], []
    for nid in nodes:
        labels = concept_labels[nid].get("pref", []) + concept_labels[nid].get("alt", [])
        if not labels: labels = [nid]
        labels = [basic_clean(x) for x in labels if x]
        texts.append(" \n ".join(labels))
        idx_map.append(nid)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    return idx_map, embs

def _labels_for_node(concept_labels: Dict[str, Dict[str, List[str]]], nid: str) -> str:
    d = concept_labels.get(nid, {})
    labels = (d.get("pref", []) or []) + (d.get("alt", []) or [])
    return " \n ".join(labels).lower()

@st.cache_data(show_spinner=True)
def build_thematic_indices(concept_labels: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    ECON_KWS = ["econom","industry","manufactur","trade","market","investment","finance","employment","gdp","sme","digital","innovation"]
    ENV_KWS  = ["environment","sustain","climate","emission","biodivers","conservation","net zero","renewable","fisheries","aquaculture","water","waste"]
    GOV_KWS  = ["govern","institution","regulation","policy","public","budget","fiscal","transparency","e-government","service"]
    SOC_KWS  = ["human","man and societ","societ","education","health","wellbeing","culture","community","youth","women","social"]

    idx = {"econ": [], "env": [], "gov": [], "soc": []}
    for nid in concept_labels.keys():
        lab = _labels_for_node(concept_labels, nid)
        if any(kw in lab for kw in ECON_KWS): idx["econ"].append(nid)
        if any(kw in lab for kw in ENV_KWS):  idx["env"].append(nid)
        if any(kw in lab for kw in GOV_KWS):  idx["gov"].append(nid)
        if any(kw in lab for kw in SOC_KWS):  idx["soc"].append(nid)
    return idx

# --------------------------- Embedding & mapping ---------------------------
def zsc_axes(sentences: List[str], zsc_pipe, threshold: float = 0.55, progress=None) -> pd.DataFrame:
    labels_en = AXES_EN
    labels_ar = ["الإنسان والمجتمع","الاقتصاد والتنمية","الْحوكَمَة والأداء المؤسسي","البيئة والاستدامة"]
    labels = labels_en + labels_ar
    rows: List[Dict[str, Any]] = []
    B = 32
    total = max(1, (len(sentences) + B - 1) // B)
    for bi, i in enumerate(range(0, len(sentences), B), start=1):
        chunk = sentences[i:i+B]
        res = zsc_pipe(chunk, candidate_labels=labels, hypothesis_template="This text is about {}.", multi_label=True)
        for s, r in zip(chunk, res):
            scores = dict(zip(r["labels"], r["scores"]))
            ax_scores = {
                AXES_EN[0]: scores.get(AXES_EN[0], 0) + scores.get(labels_ar[0], 0),
                AXES_EN[1]: scores.get(AXES_EN[1], 0) + scores.get(labels_ar[1], 0),
                AXES_EN[2]: scores.get(AXES_EN[2], 0) + scores.get(labels_ar[2], 0),
                AXES_EN[3]: scores.get(AXES_EN[3], 0) + scores.get(labels_ar[3], 0),
            }
            keep = {ax: sc for ax, sc in ax_scores.items() if sc >= threshold}
            if not keep:
                top_ax = max(ax_scores, key=ax_scores.get)
                keep = {top_ax: ax_scores[top_ax]}
            rows.append({"sentence": s, **keep})
        if progress:
            progress.progress(min(1.0, bi/total), text=f"ZSC chunks: {bi}/{total}")
    long = []
    for r in rows:
        s = r.pop("sentence")
        for ax, sc in r.items():
            long.append({"sentence": s, "axis": ax, "score": sc})
    return pd.DataFrame(long)

def embed_sentences(sentences: List[str], sbert_model_name: str, progress=None):
    model = load_sbert(sbert_model_name)
    if progress: progress.progress(0.05, text="Embedding sentences…")
    embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
    if progress: progress.progress(1.0, text="Sentence embeddings ready")
    return embs

def map_sentences_to_concepts(
    sent_embs: np.ndarray,
    idx_map: List[str],
    concept_embs: np.ndarray,
    allowed_nodes: Optional[Set[str]] = None,
    topk: int = 3,
    sim_thresh: float = 0.45,
    progress=None
) -> List[List[Tuple[str, float]]]:
    if allowed_nodes is not None:
        mask = np.array([nid in allowed_nodes for nid in idx_map])
        if not mask.any(): return [[] for _ in range(len(sent_embs))]
        filtered_embs = concept_embs[mask]
        filtered_ids  = [nid for nid, m in zip(idx_map, mask) if m]
        sims = cosine_similarity(sent_embs, filtered_embs)
        id_list = filtered_ids
    else:
        sims = cosine_similarity(sent_embs, concept_embs)
        id_list = idx_map

    res: List[List[Tuple[str, float]]] = []
    total = sims.shape[0]
    for i in range(total):
        v = sims[i]
        k = min(topk, v.shape[0])
        top_idx = np.argpartition(v, -k)[-k:]
        top_idx = top_idx[np.argsort(-v[top_idx])]
        picks = [(id_list[j], float(v[j])) for j in top_idx if float(v[j]) >= sim_thresh]
        res.append(picks)
        if progress and (i+1) % max(1, total//20) == 0:
            progress.progress((i+1)/total, text=f"Mapping sentences → concepts: {i+1}/{total}")
    if progress: progress.progress(1.0, text="Concept mapping complete")
    return res

# --------------------------- Wu–Palmer BMA ---------------------------
@st.cache_data(show_spinner=True)
def precompute_depths_and_ancestors(_G: Any) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
    G = _G  # leading underscore avoids Streamlit hashing the DiGraph
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    depth: Dict[str, int] = {n: 0 for n in G.nodes}
    from collections import deque
    dq = deque(roots)
    seen = set(roots)
    while dq:
        u = dq.popleft()
        for v in G.successors(u):
            depth[v] = max(depth.get(v, 0), depth[u] + 1)
            if v not in seen:
                seen.add(v)
                dq.append(v)
    ancestors: Dict[str, Set[str]] = {n: set(nx.ancestors(G, n)) | {n} for n in G.nodes}
    return depth, ancestors

def wup_similarity(G: Any, a: str, b: str, depth: Dict[str, int], ancestors: Dict[str, Set[str]]) -> float:
    if a not in ancestors or b not in ancestors: return 0.0
    commons = ancestors[a] & ancestors[b]
    if not commons: return 0.0
    lcs = max(commons, key=lambda n: depth.get(n, 0))
    da, db, dl = depth.get(a, 0), depth.get(b, 0), depth.get(lcs, 0)
    denom = da + db
    return (2.0 * dl / denom) if denom > 0 else 0.0

def bma_set_similarity(G: Any, A: Set[str], B: Set[str],
                       depth: Dict[str, int], ancestors: Dict[str, Set[str]],
                       progress=None, label: str = "") -> float:
    if not A or not B: return 0.0
    A_list, B_list = list(A), list(B)
    def best_avg(src: List[str], dst: List[str], tag: str) -> float:
        if not src: return 0.0
        vals, total = [], len(src)
        for i, x in enumerate(src, start=1):
            best = 0.0
            for y in dst:
                s = wup_similarity(G, x, y, depth, ancestors)
                if s > best: best = s
            vals.append(best)
            if progress and (i % max(1, total//10) == 0):
                progress.progress(i/total, text=f"{label} {tag}: {i}/{total}")
        return float(np.mean(vals)) if vals else 0.0
    return 0.5 * (best_avg(A_list, B_list, "A→B") + best_avg(B_list, A_list, "B→A"))

# --------------------------- Sidebar ---------------------------
st.sidebar.header("Inputs")
plans_zip  = st.sidebar.file_uploader("ZIP of plans (Vision 2040 + sector plans)", type=["zip"])
plan10_xls = st.sidebar.file_uploader("10th Program Plan (Excel)", type=["xlsx","xlsm","xls"])
unbis_ttl  = st.sidebar.file_uploader("UNBIS Thesaurus (TTL)", type=["ttl"])

st.sidebar.header("Models & thresholds")
zsc_model_name = st.sidebar.selectbox("ZSC model (Vision only)",
    ["joeddav/xlm-roberta-large-xnli","MoritzLaurer/multilingual-MiniLMv2-L12-mnli"], index=0)
sbert_model_name = st.sidebar.selectbox("SBERT model",
    ["paraphrase-multilingual-MiniLM-L12-v2","paraphrase-multilingual-mpnet-base-v2"], index=0)
zsc_thresh = st.sidebar.slider("ZSC axis threshold", 0.10, 0.90, 0.55, 0.01)
concept_sim_thresh = st.sidebar.slider("Concept similarity threshold", 0.10, 0.90, 0.45, 0.01)
concept_topk = st.sidebar.slider("Top-k concepts per sentence", 1, 10, 3, 1)
use_aggressive = st.sidebar.checkbox("Use aggressive cleaning (recommended)", True)

run_btn = st.sidebar.button("Run analysis", type="primary")

# --------------------------- Execute pipeline ---------------------------
if run_btn:
    if not (plans_zip and unbis_ttl and plan10_xls):
        st.error("Please upload Plans ZIP, 10th Program Plan Excel, and UNBIS TTL.")
        st.stop()

    # 1) Read plans
    with status_block("Step 1/8 – Reading plans ZIP"):
        raw_plans = read_plans_zip(plans_zip.getvalue())
        st.success(f"Loaded {len(raw_plans)} plan files with text.")

    # 2) UNBIS prep
    with status_block("Step 2/8 – Parsing UNBIS & embeddings"):
        G, concept_labels = parse_unbis_ttl(unbis_ttl.getvalue())
        idx_map, concept_embs = embed_concepts(concept_labels, sbert_model_name)
        theme_index = build_thematic_indices(concept_labels)
        depth, ancestors = precompute_depths_and_ancestors(G)
        st.success("UNBIS parsed, embeddings built, thematic indices and graph depths ready.")

    # 3) Read 10th plan Excel
    with status_block("Step 3/8 – Reading 10th Program Plan"):
        try:
            df10 = pd.read_excel(plan10_xls)
            st.success(f"Rows loaded: {len(df10)}")
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()

    # 4) Detect plans & sentence extraction (all plans)
    with status_block("Step 4/8 – Detecting plans & extracting/cleaning sentences"):
        plan_map: Dict[str,str] = {}
        for fname in raw_plans.keys():
            low = fname.lower()
            for pname, pat in PLAN_NAME_PATTERNS.items():
                if re.search(pat, low):
                    plan_map[pname] = fname
        if "Vision 2040" not in plan_map:
            st.error("Couldn't detect Vision 2040 in the ZIP.")
            st.stop()

        plan_sentences: Dict[str, List[str]] = {}
        prog = st.progress(0.0, text="Extracting…")
        det = sorted(plan_map.items())
        for i, (pname, fname) in enumerate(det, start=1):
            text = raw_plans.get(fname, "")
            sents = split_sentences(text)
            sents = [aggressive_clean(s) if use_aggressive else basic_clean(s) for s in sents]
            sents = [s for s in sents if len(s) >= 20]
            uniq = list(dict.fromkeys(sents))
            plan_sentences[pname] = uniq
            prog.progress(i/len(det), text=f"{pname}: {len(uniq)} sentences")

        # coverage (exclude NDEP summary)
        if _PX_OK and plan_sentences:
            cov = pd.Series({p: len(s) for p, s in plan_sentences.items()
                             if p != "National Digital Economy Program (summary)"}).sort_values(ascending=True)
            if not cov.empty:
                fig_cov = px.bar(cov, orientation="h",
                    labels={"value":"# Sentences","index":"Plan"},
                    title="Sentence counts by plan")
                fig_cov.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
                st.plotly_chart(fig_cov, use_container_width=True)
        st.success(f"Extracted & cleaned sentences for {len(plan_sentences)} detected plans.")

    # 5) ZSC only for Vision 2040
    with status_block("Step 5/8 – Zero-shot classification (Vision only)"):
        zsc_pipe = load_zsc(zsc_model_name)
        vision_sents = plan_sentences.get("Vision 2040", [])
        zsc_prog = st.progress(0.0, text="Starting ZSC…")
        df_axes_vision = zsc_axes(vision_sents, zsc_pipe, threshold=zsc_thresh, progress=zsc_prog)
        if _PX_OK and not df_axes_vision.empty:
            counts = df_axes_vision.groupby("axis")["sentence"].nunique().reindex(AXES_EN).fillna(0)
            fig_ax = px.bar(counts, labels={"value":"# Sentences","index":"Vision axis"},
                            title="Vision 2040 – sentences per axis")
            fig_ax.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=360)
            st.plotly_chart(fig_ax, use_container_width=True)
        st.success(f"ZSC complete: {len(df_axes_vision)} sentence-axis assignments.")

    # 6) Build 10th axis frames
    with status_block("Step 6/8 – Building 10th Program Plan axis frames"):
        axis_col_candidates = ["Axis","Pillar","المحور","محور","الفئة"]
        axis_col = next((c for c in axis_col_candidates if c in df10.columns), None)
        if axis_col is None:
            for c in df10.columns:
                v = df10[c].astype(str).str.lower().head(100).tolist()
                if any("econom" in x or "تنمي" in x for x in v):
                    axis_col = c; break
        if axis_col is None:
            st.error("Couldn't detect an axis column in the 10th Program Plan Excel.")
            st.stop()

        text_cols = [c for c in df10.columns if c != axis_col and df10[c].notna().any()]
        def row_to_text(row) -> str:
            parts = [str(row[c]) for c in text_cols if pd.notna(row.get(c))]
            joined = ". ".join(parts)
            return aggressive_clean(joined) if use_aggressive else basic_clean(joined)

        axis_frames_10: Dict[str,List[str]] = {ax: [] for ax in AXES_EN}
        prog = st.progress(0.0, text="Splitting by axis…")
        for i, (_, row) in enumerate(df10.iterrows(), start=1):
            axis_val = str(row.get(axis_col, "")).strip()
            a_low = axis_val.lower()
            if   re.search(r"human|man\s*and\s*societ|societ|مجتمع|إنسان", a_low):
                axis_match = "Man and society"
            elif re.search(r"econom|تنمي|اقتصاد", a_low):
                axis_match = "Economy and Development"
            elif re.search(r"govern|حكم|مؤس", a_low):
                axis_match = "Governance and Institutional Performance"
            elif re.search(r"enviro|بيئ|استدام", a_low):
                axis_match = "Environment and Sustainability"
            else:
                continue
            sents = [s for s in split_sentences(row_to_text(row)) if len(s) >= 20]
            axis_frames_10[axis_match].extend(sents)
            if i % max(1, len(df10)//20) == 0:
                prog.progress(i/len(df10), text=f"Processed rows: {i}/{len(df10)}")

        if _PX_OK:
            counts10 = pd.Series({ax: len(v) for ax, v in axis_frames_10.items()}).reindex(AXES_EN)
            fig10 = px.bar(counts10, labels={"value":"# Sentences","index":"10th axis"},
                           title="10th Program Plan – sentences per axis")
            fig10.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=360)
            st.plotly_chart(fig10, use_container_width=True)
        st.success("Axis-wise sentence sets built.")

    # 7) Map sentences → UNBIS concepts
    with status_block("Step 7/8 – Mapping to UNBIS concepts"):
        def concept_set_for_sentences(sents: List[str], theme: str, label: str) -> Set[str]:
            if not sents: return set()
            eprog = st.progress(0.0, text=f"{label}: embedding…")
            embs = embed_sentences(sents, sbert_model_name, progress=eprog)
            allowed = set(theme_index.get(theme, []))
            mprog = st.progress(0.0, text=f"{label}: mapping…")
            s2c = map_sentences_to_concepts(
                embs, idx_map, concept_embs,
                allowed_nodes=allowed if len(allowed)>0 else None,
                topk=concept_topk, sim_thresh=concept_sim_thresh, progress=mprog
            )
            concepts = {cid for lst in s2c for (cid, _) in lst}
            if len(concepts) == 0 and len(allowed) > 0:
                mprog = st.progress(0.0, text=f"{label}: mapping (fallback)")
                s2c = map_sentences_to_concepts(
                    embs, idx_map, concept_embs,
                    allowed_nodes=None, topk=concept_topk, sim_thresh=concept_sim_thresh, progress=mprog
                )
                concepts = {cid for lst in s2c for (cid, _) in lst}
            return concepts

        # Vision axes
        vision_axis_concepts: Dict[str, Set[str]] = {ax: set() for ax in AXES_EN}
        for ax in AXES_EN:
            sents_ax = df_axes_vision.loc[df_axes_vision["axis"] == ax, "sentence"].tolist()
            vision_axis_concepts[ax] = concept_set_for_sentences(sents_ax, AX_TO_THEME[ax], f"Vision axis: {ax}")

        # 10th axes
        axis10_concepts: Dict[str, Set[str]] = {ax: set() for ax in AXES_EN}
        for ax, sents in axis_frames_10.items():
            axis10_concepts[ax] = concept_set_for_sentences(sents, AX_TO_THEME[ax], f"10th axis: {ax}")

        # Sector plans (NDEP removed)
        sector_all_concepts: Dict[str, Set[str]] = {}
        sec_plans = sorted(set(ECO_LIST + ENV_LIST + GOV_LIST + SOC_LIST))
        prog = st.progress(0.0, text="Sector plans: embedding + mapping…")
        for i, pname in enumerate(sec_plans, start=1):
            sents = plan_sentences.get(pname, [])
            if not sents:
                sector_all_concepts[pname] = set()
            else:
                embs = embed_sentences(sents, sbert_model_name)
                # Broad
                s2c_all = map_sentences_to_concepts(embs, idx_map, concept_embs, allowed_nodes=None, topk=concept_topk, sim_thresh=concept_sim_thresh)
                bag_all = {cid for lst in s2c_all for (cid, _) in lst}
                # Robust theme unions
                union = set(bag_all)
                for theme_key in ["econ","env","gov","soc"]:
                    allowed = set(theme_index.get(theme_key, []))
                    if allowed:
                        s2c_t = map_sentences_to_concepts(embs, idx_map, concept_embs, allowed_nodes=allowed, topk=concept_topk, sim_thresh=concept_sim_thresh)
                        union |= {cid for lst in s2c_t for (cid, _) in lst}
                sector_all_concepts[pname] = union
            prog.progress(i/len(sec_plans), text=f"{pname}: mapped")
        st.success("All mappings complete.")

    # 8) Similarities
    with status_block("Step 8/8 – Computing similarities (Wu–Palmer BMA)"):
        # (A) 10th vs Vision heatmap
        df_10_vs_vision = pd.DataFrame(np.zeros((4,4)), index=AXES_EN, columns=AXES_EN)
        for i, ax10 in enumerate(AXES_EN):
            A = axis10_concepts.get(ax10, set())
            for j, axv in enumerate(AXES_EN):
                B = vision_axis_concepts.get(axv, set())
                p = st.progress(0.0, text=f"{ax10} ↔ {axv}")
                df_10_vs_vision.iloc[i,j] = bma_set_similarity(G, A, B, depth, ancestors, progress=p, label=f"{ax10} ↔ {axv}")
        st.success("Computed 10th vs Vision matrix.")

        # (B) sector vs matching 10th axis
        def compare_sector_list(sector_plans: List[str], axis_key: str, theme: str) -> pd.DataFrame:
            rows = []
            A = axis10_concepts.get(axis_key, set())
            theme_nodes = set(theme_index.get(theme, []))
            for pname in sector_plans:
                B_all = sector_all_concepts.get(pname, set())
                B_thematic = B_all & theme_nodes if theme_nodes else B_all
                B = B_thematic if len(B_thematic) > 0 else B_all
                p = st.progress(0.0, text=f"{pname} ↔ {axis_key}")
                sim = bma_set_similarity(G, A, B, depth, ancestors, progress=p, label=f"{pname} ↔ {axis_key}")
                rows.append({"Plan": pname, axis_key: sim})
            return pd.DataFrame(rows).set_index("Plan")

        econ_sim_df = compare_sector_list(ECO_LIST, "Economy and Development", "econ")
        env_sim_df  = compare_sector_list(ENV_LIST, "Environment and Sustainability", "env")
        gov_sim_df  = compare_sector_list(GOV_LIST, "Governance and Institutional Performance", "gov")
        soc_sim_df  = compare_sector_list(SOC_LIST, "Man and society", "soc")
        st.success("All similarities computed.")

    # --------------------------- Results (visuals only) ---------------------------
    st.header("Results")

    # 1) Heatmap: 10th axes vs Vision axes (order locked)
    st.subheader("10th Program Plan (axes) vs Vision 2040 (axes)")
    if _PX_OK:
        df_show = df_10_vs_vision.reindex(index=AXES_EN, columns=AXES_EN)
        fig = px.imshow(
            df_show,
            text_auto=True, aspect="auto", origin="lower",
            color_continuous_scale="Blues", zmin=0.0, zmax=1.0,
            labels=dict(color="Similarity"),
        )
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=520,
                          xaxis_title="Vision 2040 axes", yaxis_title="10th Program Plan axes",
                          font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    # 2) Sector Plans vs 10th – show ALL axes (non-interactive), with equal bar thickness across axes
    st.subheader("Sector Plans vs 10th Program Plan (per matching axis)")

    PALETTE = {"econ":["#2563eb"], "env":["#10b981"], "gov":["#7c3aed"], "soc":["#f59e0b"]}

    # same number of pixels per bar across all 4 charts
    MAX_PLANS = max(len(econ_sim_df), len(env_sim_df), len(gov_sim_df), len(soc_sim_df)) or 1
    BASE_BAR_HEIGHT = 36  # px per bar across all axes
    FIXED_HEIGHT = BASE_BAR_HEIGHT * MAX_PLANS + 140  # consistent figure height per axis

    def bars_for_sector(df: pd.DataFrame, title: str, color_seq):
        if df.empty:
            st.info(f"No data for {title}."); return
        s = df.iloc[:,0].sort_values(ascending=True)
        fig = px.bar(
            s, orientation="h",
            labels={"value":"Similarity","index":"Plan"},
            title=title, range_x=[0,1],
            color=s.index, color_discrete_sequence=color_seq,
        )
        fig.update_layout(
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10),
            height=FIXED_HEIGHT,
            xaxis=dict(title="Similarity", tickformat=".2f"),
            yaxis=dict(categoryorder="array", categoryarray=list(s.index)),
            font=dict(size=13),
        )
        st.plotly_chart(fig, use_container_width=True)

    bars_for_sector(econ_sim_df, "Economy and Development – Alignment to 10th Axis", PALETTE["econ"])
    bars_for_sector(env_sim_df,  "Environment and Sustainability – Alignment to 10th Axis", PALETTE["env"])
    bars_for_sector(gov_sim_df,  "Governance and Institutional Performance – Alignment to 10th Axis", PALETTE["gov"])
    bars_for_sector(soc_sim_df,  "Man and society – Alignment to 10th Axis", PALETTE["soc"])

    st.caption("Pipeline: aggressive cleaning → ZSC (Vision only) → SBERT embeddings → UNBIS mapping with themed gates + fallback → Wu–Palmer BMA.")
