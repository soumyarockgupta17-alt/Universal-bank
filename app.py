import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                              accuracy_score, precision_score, recall_score, f1_score)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D27"
ACCENT   = "#6C63FF"
GREEN    = "#00D4AA"
RED      = "#FF6B6B"
YELLOW   = "#FFD166"
BLUE     = "#4FC3F7"
TEXT     = "#E8EAF0"
SUBTEXT  = "#9499B0"
PALETTE  = [ACCENT, GREEN, RED, YELLOW, BLUE, "#FF9F43", "#A29BFE", "#FD79A8"]

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="IBM Plex Mono, monospace"),
        xaxis=dict(gridcolor="#2E3248", zerolinecolor="#2E3248"),
        yaxis=dict(gridcolor="#2E3248", zerolinecolor="#2E3248"),
        colorway=PALETTE,
        legend=dict(bgcolor=CARD_BG, bordercolor="#2E3248"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0F1117;
    color: #E8EAF0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13151F;
    border-right: 1px solid #2E3248;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #9499B0 !important;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #1A1D27;
    border: 1px solid #2E3248;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6C63FF; }
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #6C63FF;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #9499B0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}
.metric-delta {
    font-size: 0.8rem;
    color: #00D4AA;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6C63FF;
    border-left: 3px solid #6C63FF;
    padding-left: 10px;
    margin: 2rem 0 1rem 0;
}

/* Page title */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #E8EAF0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.page-subtitle {
    color: #9499B0;
    font-size: 0.95rem;
    margin-top: 6px;
}

/* Badge */
.badge {
    display: inline-block;
    background: #6C63FF22;
    color: #6C63FF;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    margin-right: 6px;
}

/* Rank table */
.rank-row {
    display: flex;
    align-items: center;
    background: #1A1D27;
    border: 1px solid #2E3248;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    gap: 16px;
}
.rank-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #2E3248;
    min-width: 28px;
}
.rank-name { flex: 1; font-weight: 600; font-size: 0.95rem; }
.rank-auc {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: #6C63FF;
}

/* Plotly chart containers */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #13151F; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #9499B0; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #1A1D27 !important; color: #E8EAF0 !important; border-radius: 8px; }

/* Selectbox */
.stSelectbox > div > div { background: #1A1D27; border-color: #2E3248; color: #E8EAF0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = {"Age","Experience","Income","Family","CCAvg","Education",
                 "Mortgage","Personal Loan","Securities Account","CD Account",
                 "Online","CreditCard"}

def clean_df(raw: pd.DataFrame) -> pd.DataFrame:
    raw.columns = raw.columns.str.strip()
    raw = raw.drop(columns=["ID","ZIP Code"], errors="ignore")
    raw = raw[raw["Experience"] >= 0]
    return raw.reset_index(drop=True)

def read_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

@st.cache_data
def load_base_data():
    df = pd.read_csv("UniversalBank.csv")
    return clean_df(df)

@st.cache_data
def train_models(df_hash, df):
    """Train all 7 models. df_hash makes cache key sensitive to data changes."""
    target = "Personal Loan"
    features = [c for c in df.columns if c != target]
    X, y = df[features], df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), True),
        "Naive Bayes":         (GaussianNB(), False),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=7, random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        "SVM":                 (SVC(probability=True, random_state=42), True),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, (model, scale) in models.items():
        Xtr = X_train_s if scale else X_train.values
        Xte = X_test_s  if scale else X_test.values
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        cv_s   = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")
        results[name] = dict(
            model=model, y_pred=y_pred, y_prob=y_prob,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_prob),
            cv_auc=cv_s.mean(), cv_std=cv_s.std(),
        )

    feature_importance = dict(
        zip(features, results["Random Forest"]["model"].feature_importances_))

    return results, X_test, y_test, features, feature_importance

# ── Session state: holds current working dataframe ────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = load_base_data()
if "data_source_label" not in st.session_state:
    st.session_state.data_source_label = "Default (UniversalBank.csv)"
if "upload_log" not in st.session_state:
    st.session_state.upload_log = []   # list of dicts with upload history

df = st.session_state.df

with st.spinner("🧠 Training 7 models…"):
    df_hash = str(len(df)) + str(df.iloc[0].values.tolist()) + str(df.iloc[-1].values.tolist())
    results, X_test, y_test, features, feat_imp = train_models(df_hash, df)

target = "Personal Loan"
model_names = list(results.keys())
SHORT = {
    "Logistic Regression": "LR", "K-Nearest Neighbors": "KNN",
    "Naive Bayes": "NB", "Decision Tree": "DT",
    "Random Forest": "RF", "Gradient Boosting": "GB", "SVM": "SVM",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div style='font-family: IBM Plex Mono; font-size: 1.1rem; font-weight: 700; color: #E8EAF0;'>🏦 Universal Bank</div>
        <div style='color: #9499B0; font-size: 0.78rem; margin-top: 4px;'>Personal Loan Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("Navigate", [
        "📊 Overview & EDA",
        "🤖 Model Performance",
        "📈 ROC & Thresholds",
        "🔍 Feature Analysis",
        "🔮 Predict a Customer",
        "📂 Upload Data",
    ])

    st.markdown("---")

    # ── Data source indicator ────────────────────────────────────────────────
    src_color = "#00D4AA" if st.session_state.data_source_label != "Default (UniversalBank.csv)" else "#9499B0"
    st.markdown(f"""
    <div style='font-size:0.68rem;color:#9499B0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Active Dataset</div>
    <div style='font-size:0.8rem;color:{src_color};font-family:IBM Plex Mono;word-break:break-all;'>{st.session_state.data_source_label}</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;color:#9499B0;text-transform:uppercase;letter-spacing:0.1em;margin:14px 0 8px 0;'>Dataset Info</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:IBM Plex Mono;color:#6C63FF;font-size:1.1rem;font-weight:700;'>{len(df):,}</div><div style='color:#9499B0;font-size:0.75rem;'>customers</div>", unsafe_allow_html=True)
    loan_rate = df[target].mean()*100
    st.markdown(f"<div style='font-family:IBM Plex Mono;color:#00D4AA;font-size:1.1rem;font-weight:700;margin-top:12px;'>{loan_rate:.1f}%</div><div style='color:#9499B0;font-size:0.75rem;'>loan acceptance rate</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:IBM Plex Mono;color:#FFD166;font-size:1.1rem;font-weight:700;margin-top:12px;'>{len(features)}</div><div style='color:#9499B0;font-size:0.75rem;'>features used</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#9499B0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Best Model</div>", unsafe_allow_html=True)
    best = max(results, key=lambda n: results[n]["roc_auc"])
    st.markdown(f"<div style='color:#E8EAF0;font-weight:600;'>{best}</div><div style='font-family:IBM Plex Mono;color:#6C63FF;font-size:0.9rem;'>AUC {results[best]['roc_auc']:.4f}</div>", unsafe_allow_html=True)

    if len(st.session_state.upload_log) > 0:
        st.markdown("---")
        st.markdown("<div style='font-size:0.68rem;color:#9499B0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>Upload History</div>", unsafe_allow_html=True)
        for entry in reversed(st.session_state.upload_log[-3:]):
            st.markdown(f"<div style='font-size:0.75rem;color:#9499B0;margin-bottom:4px;'>• {entry}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OFFER ENGINE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def build_offers(age, income, family, ccavg, education, mortgage,
                 sec_acc, cd_acc, online, credit_card, ensemble_prob):
    """
    Rule-based personalised offer engine.
    Returns a list of offer dicts: {title, tag, desc, rate, cta, color, icon, priority_score}
    Higher priority_score = more relevant to this customer.
    """
    edu_label = {1: "Undergrad", 2: "Graduate", 3: "Advanced"}[education]  # noqa: F841
    offers = []

    # ── 1. Core Personal Loan ──────────────────────────────────────────────
    if income >= 100:
        rate = "8.5% p.a."
        limit = f"${min(income * 5, 500):.0f}K"
        headline = "Premium Personal Loan"
        desc = f"As a high-income earner (${income}K/yr), you qualify for our premium loan tier with a higher credit limit up to {limit} and preferential rate."
    elif income >= 50:
        rate = "10.5% p.a."
        limit = f"${min(income * 4, 200):.0f}K"
        headline = "Personal Loan — Standard"
        desc = f"Based on your income profile (${income}K/yr), you are pre-approved for up to {limit} at a competitive rate."
    else:
        rate = "12.9% p.a."
        limit = f"${min(income * 3, 80):.0f}K"
        headline = "Starter Personal Loan"
        desc = f"Kickstart your financial journey with a loan up to {limit}. Simple application, fast approval."
    offers.append(dict(
        title=headline, tag="PERSONAL LOAN", desc=desc,
        rate=rate, cta="Apply Now →", color=ACCENT, icon="💳",
        priority_score=ensemble_prob * 100,
    ))

    # ── 2. Home Loan / Mortgage top-up ────────────────────────────────────
    if mortgage > 0:
        top_up = min(mortgage * 0.3, 150)
        offers.append(dict(
            title="Mortgage Top-Up Loan",
            tag="HOME LOAN",
            desc=f"You already have a mortgage of ${mortgage}K with us. Unlock up to ${top_up:.0f}K additional funds at a lower rate than an unsecured loan — no new property valuation needed.",
            rate="7.9% p.a.",
            cta="Check Eligibility →",
            color=BLUE,
            icon="🏠",
            priority_score=70 + (mortgage / 635) * 20,
        ))
    elif income >= 60 and age >= 28:
        offers.append(dict(
            title="First Home Buyer Loan",
            tag="HOME LOAN",
            desc=f"At ${income}K income and age {age}, you may qualify for our First Home Buyer package — up to ${min(income*6, 800):.0f}K with a 10% deposit option and waived processing fees.",
            rate="8.2% p.a.",
            cta="Get Pre-Approved →",
            color=BLUE,
            icon="🏠",
            priority_score=55,
        ))

    # ── 3. Credit Card upgrade ────────────────────────────────────────────
    if credit_card == 0 and ccavg >= 2:
        offers.append(dict(
            title="UniversalBank Rewards Card",
            tag="CREDIT CARD",
            desc=f"You spend ~${ccavg}K/month on cards but don't hold our card yet. Switch and earn 3× reward points on every purchase, plus a 0% balance transfer for 12 months.",
            rate="0% for 12 months",
            cta="Get the Card →",
            color=YELLOW,
            icon="✨",
            priority_score=50 + ccavg * 5,
        ))
    elif credit_card == 1 and ccavg >= 3:
        offers.append(dict(
            title="Platinum Card Upgrade",
            tag="CREDIT CARD",
            desc=f"Your spending of ${ccavg}K/month qualifies you for our Platinum tier — airport lounge access, 5× points on travel & dining, and a ${min(int(ccavg*5), 50)},000 higher credit limit.",
            rate="Existing card rate",
            cta="Upgrade Now →",
            color=YELLOW,
            icon="💎",
            priority_score=55 + ccavg * 4,
        ))

    # ── 4. Savings / Investment ───────────────────────────────────────────
    if sec_acc == 0 and income >= 70:
        offers.append(dict(
            title="Securities & Investments Account",
            desc=f"Grow your wealth. At ${income}K income you have surplus capacity to invest. Open a securities account with zero brokerage for the first 6 months and access to 3,000+ listed securities.",
            tag="INVESTMENTS",
            rate="0% brokerage · 6 months",
            cta="Start Investing →",
            color=GREEN,
            icon="📈",
            priority_score=40 + (income / 224) * 20,
        ))

    if cd_acc == 0:
        cd_rate = "5.8%" if income >= 80 else "5.2%"
        offers.append(dict(
            title="Certificate of Deposit",
            tag="SAVINGS",
            desc=f"Lock in a guaranteed {cd_rate} p.a. return. Ideal for parking funds you won't need for 12–24 months. FDIC insured, no market risk.",
            rate=f"{cd_rate} p.a. guaranteed",
            cta="Open CD Account →",
            color=GREEN,
            icon="🏦",
            priority_score=35,
        ))

    # ── 5. Digital / Online banking ───────────────────────────────────────
    if online == 0:
        offers.append(dict(
            title="Go Digital — Bonus Offer",
            tag="DIGITAL BANKING",
            desc="Activate online banking and get $50 cashback on your first digital transaction, plus fee-free NEFT/RTGS transfers for 6 months. Manage everything from your phone.",
            rate="$50 cashback",
            cta="Activate Now →",
            color="#A29BFE",
            icon="📱",
            priority_score=30,
        ))

    # ── 6. Family / Education loan ────────────────────────────────────────
    if family >= 3 and education <= 2:
        offers.append(dict(
            title="Family Education Loan",
            tag="EDUCATION LOAN",
            desc=f"With a family of {family}, education planning is key. Get up to $50K at a subsidised rate to fund higher education — deferred repayment until course completion.",
            rate="6.9% p.a. subsidised",
            cta="Learn More →",
            color="#FD79A8",
            icon="🎓",
            priority_score=38,
        ))

    # ── 7. Retirement / Wealth planning ──────────────────────────────────
    if age >= 45 and income >= 80:
        offers.append(dict(
            title="Wealth Management Advisory",
            tag="WEALTH",
            desc=f"At {age} with ${income}K income, retirement planning is critical. Our advisors can build a customised portfolio blending fixed income, equities, and tax-efficient instruments.",
            rate="Free first consultation",
            cta="Book a Session →",
            color="#FF9F43",
            icon="🌟",
            priority_score=45,
        ))

    # Sort by priority descending
    offers.sort(key=lambda o: o["priority_score"], reverse=True)
    return offers


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown("""
    <div class='page-title'>Exploratory Data Analysis</div>
    <div class='page-subtitle'>Understanding the Universal Bank customer dataset</div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown("<div class='section-header'>Key Statistics</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (f"{len(df):,}", "Total Customers", f"{df[target].sum():,} accepted loan"),
        (f"${df['Income'].mean():.0f}K", "Avg Income", f"Max ${df['Income'].max()}K"),
        (f"{df['Age'].mean():.0f}", "Avg Age", f"Range {df['Age'].min()}–{df['Age'].max()}"),
        (f"{df['CCAvg'].mean():.2f}K", "Avg CC Spend", "per month"),
        (f"{df[target].mean()*100:.1f}%", "Loan Rate", "accepted campaign"),
    ]
    for col, (val, label, delta) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-delta'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Distribution Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Target donut
    with c1:
        counts = df[target].value_counts()
        fig = go.Figure(go.Pie(
            labels=["No Loan", "Accepted Loan"],
            values=counts.values,
            hole=0.65,
            marker=dict(colors=[SUBTEXT, ACCENT], line=dict(color=DARK_BG, width=3)),
            textinfo="label+percent",
            textfont=dict(color=TEXT, size=12),
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Personal Loan Distribution",
                          annotations=[dict(text=f"<b>{counts[1]}</b><br>accepted", x=0.5, y=0.5,
                                            font=dict(size=16, color=TEXT), showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    # Income histogram
    with c2:
        fig = go.Figure()
        for val, color, label in [(0, SUBTEXT, "No Loan"), (1, ACCENT, "Accepted")]:
            fig.add_trace(go.Histogram(
                x=df[df[target]==val]["Income"], name=label,
                marker_color=color, opacity=0.75, nbinsx=35,
                marker_line=dict(width=0),
            ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Income Distribution by Loan Status",
                          xaxis_title="Annual Income ($000)",
                          barmode="overlay", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4, c5 = st.columns(3)

    # Education acceptance rate
    with c3:
        edu_rate = df.groupby("Education")[target].mean().reset_index()
        edu_rate["Education"] = edu_rate["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
        fig = px.bar(edu_rate, x="Education", y=target,
                     color="Education", color_discrete_sequence=PALETTE,
                     text=edu_rate[target].apply(lambda v: f"{v*100:.1f}%"))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Acceptance Rate by Education",
                          yaxis_title="Acceptance Rate", showlegend=False)
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Family size
    with c4:
        fam_rate = df.groupby("Family")[target].mean().reset_index()
        fig = px.bar(fam_rate, x="Family", y=target,
                     color=target, color_continuous_scale=[[0, CARD_BG],[1, YELLOW]],
                     text=fam_rate[target].apply(lambda v: f"{v*100:.1f}%"))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Acceptance by Family Size",
                          xaxis_title="Family Size", yaxis_title="Acceptance Rate",
                          showlegend=False, coloraxis_showscale=False)
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # CCAvg vs loan
    with c5:
        fig = go.Figure()
        for val, color, label in [(0, SUBTEXT, "No Loan"), (1, GREEN, "Accepted")]:
            fig.add_trace(go.Box(
                y=df[df[target]==val]["CCAvg"], name=label,
                marker_color=color, line_color=color,
                fillcolor=color+"44", boxmean=True,
            ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="CC Spending vs Loan",
                          yaxis_title="CCAvg ($000/month)")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'>Feature Correlations</div>", unsafe_allow_html=True)
    corr = df.corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, RED],[0.5, CARD_BG],[1, ACCENT]],
        zmid=0, text=corr.values.round(2),
        texttemplate="%{text}", textfont=dict(size=10, color=TEXT),
        hoverongaps=False,
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                      title="Correlation Matrix – All Features",
                      height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class='page-title'>Model Performance</div>
    <div class='page-subtitle'>Comparing 7 classifiers on test set & 5-fold cross-validation</div>
    """, unsafe_allow_html=True)

    # Summary metrics bar chart
    st.markdown("<div class='section-header'>Metrics Comparison</div>", unsafe_allow_html=True)
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    metric_labels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    short_names = [SHORT[n] for n in model_names]

    fig = go.Figure()
    for metric, label, color in zip(metrics, metric_labels, PALETTE):
        vals = [results[n][metric] for n in model_names]
        fig.add_trace(go.Bar(name=label, x=short_names, y=vals, marker_color=color,
                              marker_line_width=0, opacity=0.87))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                      barmode="group", title="All Metrics by Model",
                      yaxis=dict(range=[0.4, 1.02], gridcolor="#2E3248"),
                      height=420, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    sel_model = st.selectbox("Select model", model_names, index=4)

    c1, c2 = st.columns([1, 1])
    with c1:
        cm = confusion_matrix(y_test, results[sel_model]["y_pred"])
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                         color_continuous_scale=[[0, CARD_BG],[1, ACCENT]],
                         labels=dict(x="Predicted", y="Actual"),
                         x=["No Loan","Loan"], y=["No Loan","Loan"])
        fig.update_traces(textfont=dict(size=20, color=TEXT))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title=f"Confusion Matrix – {sel_model}",
                          coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        r = results[sel_model]
        tn, fp, fn, tp = confusion_matrix(y_test, r["y_pred"]).ravel()
        st.markdown("<div class='section-header' style='margin-top:0'>Derived Metrics</div>", unsafe_allow_html=True)
        stats = [
            ("Accuracy",  f"{r['accuracy']:.4f}"),
            ("Precision", f"{r['precision']:.4f}"),
            ("Recall",    f"{r['recall']:.4f}"),
            ("F1 Score",  f"{r['f1']:.4f}"),
            ("ROC-AUC",   f"{r['roc_auc']:.4f}"),
            ("CV-AUC",    f"{r['cv_auc']:.4f} ± {r['cv_std']:.3f}"),
            ("True Pos",  f"{tp}"),
            ("False Pos", f"{fp}"),
            ("False Neg", f"{fn}"),
            ("True Neg",  f"{tn}"),
        ]
        for i in range(0, len(stats), 2):
            cc1, cc2 = st.columns(2)
            for col, (label, val) in zip([cc1, cc2], stats[i:i+2]):
                col.markdown(f"""
                <div class='metric-card' style='padding:14px;'>
                    <div class='metric-value' style='font-size:1.5rem;'>{val}</div>
                    <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

    # Ranking
    st.markdown("<div class='section-header'>Model Ranking by ROC-AUC</div>", unsafe_allow_html=True)
    ranking = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    rank_colors = [ACCENT, GREEN, YELLOW, BLUE, RED, "#FF9F43", SUBTEXT]
    for rank, (name, r) in enumerate(ranking, 1):
        col_hex = rank_colors[rank-1]
        bar_pct = r["roc_auc"] * 100
        st.markdown(f"""
        <div style='background:#1A1D27;border:1px solid #2E3248;border-radius:10px;
                    padding:14px 20px;margin-bottom:8px;display:flex;align-items:center;gap:16px;'>
            <div style='font-family:IBM Plex Mono;font-size:1.4rem;font-weight:700;
                        color:{col_hex};min-width:30px;'>#{rank}</div>
            <div style='flex:1;'>
                <div style='font-weight:600;font-size:0.95rem;'>{name}</div>
                <div style='background:#0F1117;border-radius:4px;height:6px;margin-top:8px;'>
                    <div style='background:{col_hex};height:6px;border-radius:4px;width:{bar_pct:.1f}%;'></div>
                </div>
            </div>
            <div style='font-family:IBM Plex Mono;font-size:0.9rem;color:{col_hex};min-width:100px;text-align:right;'>
                AUC {r['roc_auc']:.4f}
            </div>
            <div style='font-family:IBM Plex Mono;font-size:0.85rem;color:#9499B0;min-width:70px;text-align:right;'>
                F1 {r['f1']:.3f}
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROC & THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ROC & Thresholds":
    st.markdown("""
    <div class='page-title'>ROC Curves & Threshold Analysis</div>
    <div class='page-subtitle'>Explore the precision-recall trade-off for each model</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ROC overlay
    with c1:
        st.markdown("<div class='section-header'>ROC Curves – All Models</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color=SUBTEXT, width=1),
                                  showlegend=False))
        for i, name in enumerate(model_names):
            fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
            auc = results[name]["roc_auc"]
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{SHORT[name]} ({auc:.3f})",
                line=dict(color=PALETTE[i], width=2.5),
            ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="ROC – All Models",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          height=430,
                          legend=dict(orientation="v", bgcolor=CARD_BG))
        st.plotly_chart(fig, use_container_width=True)

    # CV AUC
    with c2:
        st.markdown("<div class='section-header'>Cross-Validation AUC (5-Fold)</div>", unsafe_allow_html=True)
        names_rev = model_names[::-1]
        cv_aucs = [results[n]["cv_auc"] for n in names_rev]
        cv_stds = [results[n]["cv_std"] for n in names_rev]
        short_rev = [SHORT[n] for n in names_rev]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=short_rev, x=cv_aucs,
            orientation="h",
            marker=dict(color=PALETTE[:len(names_rev)][::-1], line_width=0),
            error_x=dict(type="data", array=cv_stds, color=TEXT, thickness=1.5, width=4),
            text=[f"{v:.3f}±{s:.3f}" for v,s in zip(cv_aucs, cv_stds)],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="5-Fold CV ROC-AUC ± Std",
                          xaxis=dict(range=[0.75, 1.05], gridcolor="#2E3248"),
                          height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Threshold exploration
    st.markdown("<div class='section-header'>Threshold Explorer</div>", unsafe_allow_html=True)
    tc1, tc2 = st.columns([1, 2])
    with tc1:
        thresh_model = st.selectbox("Model for threshold analysis", model_names, index=4)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)

    y_prob = results[thresh_model]["y_prob"]
    y_thresh = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_thresh)
    tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1_t = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    with tc2:
        t1, t2, t3, t4 = st.columns(4)
        for col, val, label, color in [
            (t1, f"{tp}", "True Positives", GREEN),
            (t2, f"{fp}", "False Positives", RED),
            (t3, f"{fn}", "False Negatives", YELLOW),
            (t4, f"{f1_t:.3f}", "F1 @ Threshold", ACCENT),
        ]:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color};'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    thresholds_arr = np.linspace(0.01, 0.99, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds_arr:
        yp = (y_prob >= t).astype(int)
        pr = precision_score(y_test, yp, zero_division=0)
        rc = recall_score(y_test, yp, zero_division=0)
        precs.append(pr); recs.append(rc)
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds_arr, y=precs, mode="lines",
                              name="Precision", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=thresholds_arr, y=recs, mode="lines",
                              name="Recall", line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=thresholds_arr, y=f1s, mode="lines",
                              name="F1", line=dict(color=YELLOW, width=2)))
    fig.add_vline(x=threshold, line=dict(color=RED, dash="dash", width=1.5),
                  annotation_text=f"t={threshold}", annotation_font_color=RED)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                      title=f"Precision / Recall / F1 vs Threshold — {thresh_model}",
                      xaxis_title="Threshold", yaxis_title="Score", height=380)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.markdown("""
    <div class='page-title'>Feature Analysis</div>
    <div class='page-subtitle'>What drives personal loan acceptance?</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # RF Feature importance
    with c1:
        st.markdown("<div class='section-header'>Random Forest – Feature Importance</div>", unsafe_allow_html=True)
        fi = pd.Series(feat_imp).sort_values()
        colors_fi = [GREEN if i == fi.index[-1] else ACCENT for i in fi.index]
        fig = go.Figure(go.Bar(
            y=fi.index, x=fi.values, orientation="h",
            marker=dict(color=colors_fi, line_width=0),
            text=[f"{v:.4f}" for v in fi.values],
            textposition="outside", textfont=dict(color=TEXT, size=10),
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Feature Importance (Random Forest)",
                          xaxis_title="Importance", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: income vs CCAvg colored by loan
    with c2:
        st.markdown("<div class='section-header'>Income vs CC Spending</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="Income", y="CCAvg",
                         color=target, color_discrete_map={0: SUBTEXT, 1: ACCENT},
                         opacity=0.6, size_max=5,
                         labels={"Income":"Income ($000)", "CCAvg":"CC Avg ($000/mo)",
                                 target: "Loan"})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title="Income vs CC Spending (colored by loan)", height=420)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

    # Feature distributions deep-dive
    st.markdown("<div class='section-header'>Feature Deep-Dive</div>", unsafe_allow_html=True)
    feat_sel = st.selectbox("Choose feature", features, index=features.index("Income"))

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for val, color, label in [(0, SUBTEXT, "No Loan"), (1, ACCENT, "Accepted")]:
            fig.add_trace(go.Histogram(
                x=df[df[target]==val][feat_sel], name=label,
                marker_color=color, opacity=0.75, nbinsx=30, marker_line_width=0,
            ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          barmode="overlay", title=f"{feat_sel} Distribution",
                          xaxis_title=feat_sel, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Mean value of feature by loan group
        group_stats = df.groupby(target)[feat_sel].describe().T.reset_index()
        fig = go.Figure()
        for val, color, label in [(0, SUBTEXT, "No Loan"), (1, ACCENT, "Accepted")]:
            sub = df[df[target]==val][feat_sel]
            fig.add_trace(go.Violin(
                y=sub, name=label, marker_color=color,
                line_color=color, fillcolor=color+"44",
                box_visible=True, meanline_visible=True,
            ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title=f"{feat_sel} Violin Plot",
                          yaxis_title=feat_sel, height=340)
        st.plotly_chart(fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT A CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Customer":
    st.markdown("""
    <div class='page-title'>Predict & Prescribe</div>
    <div class='page-subtitle'>Predict loan interest then receive a personalised offer bundle — tailored to this customer's profile</div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age       = st.slider("Age", 18, 75, 35)
        experience= st.slider("Experience (years)", 0, 50, 10)
        income    = st.slider("Annual Income ($000)", 8, 224, 80)
        family    = st.slider("Family Size", 1, 4, 2)
    with c2:
        ccavg     = st.slider("CC Avg Spend ($000/mo)", 0.0, 10.0, 2.0, 0.1)
        education = st.selectbox("Education", [1,2,3], format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
        mortgage  = st.slider("Mortgage ($000)", 0, 635, 0)
    with c3:
        sec_acc   = st.selectbox("Securities Account", [0,1], format_func=lambda x: "Yes" if x else "No")
        cd_acc    = st.selectbox("CD Account", [0,1], format_func=lambda x: "Yes" if x else "No")
        online    = st.selectbox("Online Banking", [0,1], format_func=lambda x: "Yes" if x else "No")
        credit_card = st.selectbox("UniversalBank CreditCard", [0,1], format_func=lambda x: "Yes" if x else "No")

    customer = pd.DataFrame([[age, experience, income, family, ccavg, education,
                               mortgage, sec_acc, cd_acc, online, credit_card]],
                             columns=features)

    scaler = StandardScaler()
    scaler.fit(df[features])
    customer_s = scaler.transform(customer)

    # ── Compute all model predictions ──────────────────────────────────────
    probs = []
    for name in model_names:
        uses_scale = name in ["Logistic Regression","K-Nearest Neighbors","SVM"]
        inp = customer_s if uses_scale else customer.values
        probs.append(results[name]["model"].predict_proba(inp)[0][1])
    ensemble_prob = float(np.mean(probs))
    ensemble_pred = ensemble_prob >= 0.5

    # ── Model verdict row ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Predictions</div>", unsafe_allow_html=True)
    cols = st.columns(len(model_names))
    for col, (name, prob) in zip(cols, zip(model_names, probs)):
        pred = int(prob >= 0.5)
        color = GREEN if pred == 1 else RED
        verdict = "ACCEPT" if pred == 1 else "DECLINE"
        col.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:0.7rem;color:#9499B0;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:8px;'>{SHORT[name]}</div>
            <div style='font-family:IBM Plex Mono;font-size:1.8rem;font-weight:700;
                        color:{color};'>{prob*100:.0f}%</div>
            <div style='font-size:0.7rem;color:{color};margin-top:6px;
                        font-weight:600;letter-spacing:0.1em;'>{verdict}</div>
        </div>""", unsafe_allow_html=True)

    # ── Ensemble gauge ─────────────────────────────────────────────────────
    gauge_col, verdict_col = st.columns([2, 1])
    with gauge_col:
        st.markdown("<div class='section-header'>Ensemble Probability</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ensemble_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Loan Acceptance Probability (7-Model Average)",
                   "font": {"color": TEXT, "size": 14}},
            number={"suffix": "%", "font": {"color": TEXT, "size": 40, "family":"IBM Plex Mono"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=SUBTEXT, tickfont=dict(color=SUBTEXT)),
                bar=dict(color=ACCENT if ensemble_pred else RED),
                bgcolor=CARD_BG,
                bordercolor="#2E3248",
                steps=[
                    dict(range=[0,30], color="#FF6B6B22"),
                    dict(range=[30,60], color="#FFD16622"),
                    dict(range=[60,100], color="#00D4AA22"),
                ],
                threshold=dict(line=dict(color=GREEN, width=3), thickness=0.85, value=50),
            )
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          height=320, paper_bgcolor=DARK_BG)
        st.plotly_chart(fig, use_container_width=True)

    with verdict_col:
        st.markdown("<div class='section-header'>Verdict</div>", unsafe_allow_html=True)
        if ensemble_pred:
            tier = "HIGH" if ensemble_prob >= 0.75 else "MEDIUM"
            tier_color = GREEN if tier == "HIGH" else YELLOW
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#00D4AA11,#00D4AA22);
                        border:1px solid #00D4AA55;border-radius:14px;padding:28px 20px;
                        text-align:center;margin-top:8px;'>
                <div style='font-size:2.2rem;margin-bottom:8px;'>🎯</div>
                <div style='font-family:IBM Plex Mono;font-size:1.4rem;font-weight:700;
                            color:{GREEN};'>INTERESTED</div>
                <div style='font-size:0.75rem;color:#9499B0;margin-top:6px;'>Confidence</div>
                <div style='font-family:IBM Plex Mono;color:{tier_color};font-size:1rem;
                            font-weight:700;'>{ensemble_prob*100:.1f}% · {tier}</div>
                <div style='margin-top:14px;font-size:0.78rem;color:#9499B0;line-height:1.5;'>
                    Personalised offers have been<br>generated below ↓
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#FF6B6B11,#FF6B6B22);
                        border:1px solid #FF6B6B55;border-radius:14px;padding:28px 20px;
                        text-align:center;margin-top:8px;'>
                <div style='font-size:2.2rem;margin-bottom:8px;'>🔴</div>
                <div style='font-family:IBM Plex Mono;font-size:1.4rem;font-weight:700;
                            color:{RED};'>NOT INTERESTED</div>
                <div style='font-size:0.75rem;color:#9499B0;margin-top:6px;'>Confidence</div>
                <div style='font-family:IBM Plex Mono;color:{RED};font-size:1rem;font-weight:700;'>
                    {(1-ensemble_prob)*100:.1f}% · LOW PRIORITY</div>
                <div style='margin-top:14px;font-size:0.78rem;color:#9499B0;line-height:1.5;'>
                    Consider nurture campaigns<br>or alternative products ↓
                </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # PERSONALISED OFFERS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>Personalised Offer Recommendations</div>", unsafe_allow_html=True)

    offers = build_offers(age, income, family, ccavg, education, mortgage,
                          sec_acc, cd_acc, online, credit_card, ensemble_prob)

    if ensemble_pred:
        st.markdown(f"""
        <div style='background:#6C63FF11;border:1px solid #6C63FF44;border-radius:10px;
                    padding:14px 18px;margin-bottom:18px;font-size:0.85rem;color:#9499B0;'>
            🎯 <b style='color:#E8EAF0;'>This customer is likely interested.</b>
            Below are <b style='color:#6C63FF;'>{len(offers)} personalised offers</b> ranked by relevance to their profile —
            income <b style='color:#E8EAF0;'>${income}K</b>,
            family size <b style='color:#E8EAF0;'>{family}</b>,
            CC spend <b style='color:#E8EAF0;'>${ccavg}K/mo</b>,
            education <b style='color:#E8EAF0;'>{["","Undergrad","Graduate","Advanced"][education]}</b>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#FF6B6B11;border:1px solid #FF6B6B44;border-radius:10px;
                    padding:14px 18px;margin-bottom:18px;font-size:0.85rem;color:#9499B0;'>
            🔴 <b style='color:#E8EAF0;'>Low personal loan interest predicted.</b>
            Consider these <b style='color:#FF6B6B;'>{len(offers)} alternative product offers</b> to still engage and cross-sell this customer.
        </div>""", unsafe_allow_html=True)

    # Offers grid — 2 columns
    for i in range(0, len(offers), 2):
        row_offers = offers[i:i+2]
        cols_offer = st.columns(len(row_offers))
        for col, offer in zip(cols_offer, row_offers):
            priority_pct = min(offer["priority_score"], 100)
            is_top = (i == 0 and col == cols_offer[0])
            border = f"2px solid {offer['color']}88" if is_top else f"1px solid {offer['color']}44"
            glow   = f"box-shadow:0 0 18px {offer['color']}22;" if is_top else ""
            badge  = "<span style='background:#FFD16633;color:#FFD166;border-radius:4px;padding:1px 7px;font-size:0.68rem;font-weight:700;margin-left:6px;'>⭐ TOP PICK</span>" if is_top else ""
            col.markdown(f"""
            <div style='background:#1A1D27;border:{border};border-radius:14px;
                        padding:22px 20px;height:100%;{glow}'>
                <!-- Header row -->
                <div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:14px;'>
                    <div style='font-size:1.8rem;line-height:1;'>{offer['icon']}</div>
                    <div style='flex:1;'>
                        <div style='display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:4px;'>
                            <span style='background:{offer["color"]}22;color:{offer["color"]};
                                         border:1px solid {offer["color"]}55;border-radius:4px;
                                         padding:1px 8px;font-size:0.65rem;font-family:IBM Plex Mono;
                                         letter-spacing:0.08em;font-weight:700;'>
                                {offer['tag']}
                            </span>{badge}
                        </div>
                        <div style='font-size:1rem;font-weight:700;color:#E8EAF0;line-height:1.2;'>
                            {offer['title']}
                        </div>
                    </div>
                </div>
                <!-- Description -->
                <div style='font-size:0.82rem;color:#9499B0;line-height:1.6;margin-bottom:16px;'>
                    {offer['desc']}
                </div>
                <!-- Rate + relevance bar -->
                <div style='border-top:1px solid #2E3248;padding-top:14px;'>
                    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;'>
                        <div>
                            <div style='font-size:0.65rem;color:#9499B0;text-transform:uppercase;
                                        letter-spacing:0.08em;'>Rate / Offer</div>
                            <div style='font-family:IBM Plex Mono;color:{offer["color"]};
                                        font-size:0.95rem;font-weight:700;margin-top:2px;'>
                                {offer['rate']}
                            </div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:0.65rem;color:#9499B0;text-transform:uppercase;
                                        letter-spacing:0.08em;'>Relevance</div>
                            <div style='font-family:IBM Plex Mono;color:{offer["color"]};
                                        font-size:0.9rem;font-weight:700;margin-top:2px;'>
                                {min(priority_pct,100):.0f}%
                            </div>
                        </div>
                    </div>
                    <!-- Progress bar -->
                    <div style='background:#0F1117;border-radius:4px;height:5px;margin-bottom:14px;'>
                        <div style='background:{offer["color"]};height:5px;border-radius:4px;
                                    width:{min(priority_pct,100):.0f}%;
                                    transition:width 0.6s ease;'></div>
                    </div>
                    <!-- CTA button -->
                    <div style='background:{offer["color"]}22;border:1px solid {offer["color"]}66;
                                border-radius:8px;padding:9px;text-align:center;
                                font-weight:700;font-size:0.82rem;color:{offer["color"]};
                                letter-spacing:0.04em;cursor:pointer;'>
                        {offer['cta']}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Agent talking points ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Agent Talking Points</div>", unsafe_allow_html=True)
    top_offer = offers[0]
    edu_label = {1:"Undergrad",2:"Graduate",3:"Advanced"}[education]

    talking_points = []
    if income >= 100:
        talking_points.append(f"✦ Acknowledge high-income status — position as exclusive/premium tier access")
    if mortgage > 0:
        talking_points.append(f"✦ They already have a mortgage with us — leverage existing relationship, lower perceived risk")
    if ccavg >= 3:
        talking_points.append(f"✦ High CC spend (${ccavg}K/mo) signals active financial life — highlight rewards & lifestyle benefits")
    if family >= 3:
        talking_points.append(f"✦ Family of {family} — emphasise financial security, education planning, and family protection products")
    if sec_acc == 0 and income >= 70:
        talking_points.append(f"✦ No securities account yet — strong upsell opportunity for investments")
    if online == 0:
        talking_points.append(f"✦ Not on digital banking — demo the app with a cashback incentive to onboard them digitally")
    if ensemble_pred:
        talking_points.append(f"✦ Models are {ensemble_prob*100:.0f}% confident — open with the personal loan offer first, then cross-sell")
    else:
        talking_points.append(f"✦ Low loan interest predicted — lead with a softer product (CD/savings) to build trust before pitching loan")

    for pt in talking_points:
        st.markdown(f"""
        <div style='background:#1A1D27;border-left:3px solid #6C63FF;border-radius:0 8px 8px 0;
                    padding:11px 16px;margin-bottom:8px;font-size:0.85rem;color:#E8EAF0;'>
            {pt}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Upload Data":
    st.markdown("""
    <div class='page-title'>Upload New Data</div>
    <div class='page-subtitle'>Add or replace customer data — supports CSV and Excel (.xlsx)</div>
    """, unsafe_allow_html=True)

    # ── Required columns info ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Required Columns</div>", unsafe_allow_html=True)
    col_info = [
        ("Age", "Customer age in years"),
        ("Experience", "Years of professional experience"),
        ("Income", "Annual income ($000)"),
        ("Family", "Family size (1–4)"),
        ("CCAvg", "Avg monthly CC spend ($000)"),
        ("Education", "1=Undergrad, 2=Graduate, 3=Advanced"),
        ("Mortgage", "Mortgage value ($000)"),
        ("Personal Loan", "Target: 0=No, 1=Yes"),
        ("Securities Account", "0=No, 1=Yes"),
        ("CD Account", "0=No, 1=Yes"),
        ("Online", "0=No, 1=Yes"),
        ("CreditCard", "0=No, 1=Yes"),
    ]
    cols_display = st.columns(3)
    for i, (col_name, desc) in enumerate(col_info):
        is_target = col_name == "Personal Loan"
        border_color = ACCENT if is_target else "#2E3248"
        badge_color  = ACCENT if is_target else SUBTEXT
        cols_display[i % 3].markdown(f"""
        <div style='background:#1A1D27;border:1px solid {border_color};border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;'>
            <div style='font-family:IBM Plex Mono;font-size:0.8rem;color:{badge_color};
                        font-weight:600;'>{col_name}{"  ★" if is_target else ""}</div>
            <div style='font-size:0.75rem;color:#9499B0;margin-top:3px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1A1D2788;border:1px solid #2E3248;border-radius:8px;padding:12px 16px;
                font-size:0.8rem;color:#9499B0;margin:12px 0;'>
        💡 <b style='color:#E8EAF0;'>Tips:</b>
        Columns <code>ID</code> and <code>ZIP Code</code> are automatically dropped if present.
        Rows with negative <code>Experience</code> are cleaned automatically.
        Column names are matched case-insensitively and whitespace-trimmed.
    </div>
    """, unsafe_allow_html=True)

    # ── Upload widget ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Upload File</div>", unsafe_allow_html=True)

    u1, u2 = st.columns([2, 1])
    with u1:
        uploaded = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            help="Must contain the required columns listed above."
        )
    with u2:
        mode = st.radio(
            "How to apply the new data",
            ["➕ Append to existing data", "🔄 Replace existing data"],
            help="Append adds new rows to the current dataset. Replace starts fresh with only the new file."
        )
        st.markdown(f"""
        <div style='background:#1A1D27;border:1px solid #2E3248;border-radius:8px;
                    padding:10px 14px;margin-top:8px;font-size:0.78rem;'>
            <div style='color:#9499B0;margin-bottom:4px;'>Current dataset</div>
            <div style='font-family:IBM Plex Mono;color:#6C63FF;font-size:1.1rem;font-weight:700;'>{len(df):,}</div>
            <div style='color:#9499B0;font-size:0.72rem;'>rows</div>
        </div>""", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            raw = read_upload(uploaded)
            raw.columns = raw.columns.str.strip()

            # Validate required columns
            missing = REQUIRED_COLS - set(raw.columns)
            if missing:
                st.error(f"❌ Missing required columns: **{', '.join(sorted(missing))}**")
                st.stop()

            new_df = clean_df(raw)

            # Preview
            st.markdown("<div class='section-header'>Preview — Uploaded File</div>", unsafe_allow_html=True)
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.6rem;'>{len(new_df):,}</div><div class='metric-label'>Rows</div></div>", unsafe_allow_html=True)
            p2.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.6rem;'>{len(new_df.columns)}</div><div class='metric-label'>Columns</div></div>", unsafe_allow_html=True)
            loan_pct = new_df["Personal Loan"].mean()*100
            p3.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.6rem;color:{GREEN};'>{loan_pct:.1f}%</div><div class='metric-label'>Loan Rate</div></div>", unsafe_allow_html=True)
            dupes = new_df.duplicated().sum()
            p4.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.6rem;color:{'#FF6B6B' if dupes>0 else GREEN};'>{dupes}</div><div class='metric-label'>Duplicates</div></div>", unsafe_allow_html=True)

            st.dataframe(
                new_df.head(10).style.set_properties(**{
                    "background-color": "#1A1D27",
                    "color": "#E8EAF0",
                    "border": "1px solid #2E3248",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Null check
            nulls = new_df.isnull().sum()
            if nulls.any():
                st.warning(f"⚠️ Null values detected — they will be dropped on apply: {nulls[nulls>0].to_dict()}")
                new_df = new_df.dropna()

            # ── Diff comparison if appending ─────────────────────────────────
            if "Append" in mode:
                combined = pd.concat([st.session_state.df, new_df], ignore_index=True).drop_duplicates()
                added = len(combined) - len(st.session_state.df)
                result_df = combined
                action_label = f"Append {len(new_df):,} rows (+{added} unique)"
                result_rows = len(combined)
                btn_color = GREEN
            else:
                result_df = new_df
                action_label = f"Replace with {len(new_df):,} rows"
                result_rows = len(new_df)
                btn_color = RED

            # Side-by-side before/after
            st.markdown("<div class='section-header'>Before → After</div>", unsafe_allow_html=True)
            ba1, ba2 = st.columns(2)
            with ba1:
                st.markdown(f"""
                <div class='metric-card' style='text-align:left;'>
                    <div style='font-size:0.7rem;color:#9499B0;letter-spacing:0.1em;text-transform:uppercase;'>Current</div>
                    <div style='font-family:IBM Plex Mono;font-size:1.8rem;font-weight:700;color:#9499B0;margin:4px 0;'>{len(st.session_state.df):,} rows</div>
                    <div style='font-size:0.8rem;color:#9499B0;'>Loan rate: {st.session_state.df["Personal Loan"].mean()*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with ba2:
                st.markdown(f"""
                <div class='metric-card' style='text-align:left;border-color:{btn_color}44;'>
                    <div style='font-size:0.7rem;color:#9499B0;letter-spacing:0.1em;text-transform:uppercase;'>After Apply</div>
                    <div style='font-family:IBM Plex Mono;font-size:1.8rem;font-weight:700;color:{btn_color};margin:4px 0;'>{result_rows:,} rows</div>
                    <div style='font-size:0.8rem;color:#9499B0;'>Loan rate: {result_df["Personal Loan"].mean()*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"✅ Apply — {action_label}", use_container_width=True, type="primary"):
                st.session_state.df = result_df
                st.session_state.data_source_label = (
                    f"{'+ ' if 'Append' in mode else ''}{uploaded.name} ({result_rows:,} rows)"
                )
                from datetime import datetime
                ts = datetime.now().strftime("%H:%M:%S")
                action_word = "Appended" if "Append" in mode else "Replaced"
                st.session_state.upload_log.append(
                    f"{ts} · {action_word} · {uploaded.name}"
                )
                train_models.clear()   # bust model cache so they retrain on new data
                st.success(f"✅ Dataset updated! Models will retrain automatically. Navigate to any page to see updated results.")
                st.balloons()

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    else:
        # Empty state
        st.markdown("""
        <div style='background:#1A1D27;border:2px dashed #2E3248;border-radius:12px;
                    padding:48px;text-align:center;margin-top:12px;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>📂</div>
            <div style='font-size:1rem;color:#9499B0;'>Upload a CSV or Excel file above</div>
            <div style='font-size:0.8rem;color:#6C63FF;margin-top:8px;'>
                Must include all required columns · ID and ZIP Code are auto-dropped
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Reset to default ─────────────────────────────────────────────────────
    if st.session_state.data_source_label != "Default (UniversalBank.csv)":
        st.markdown("---")
        st.markdown("<div class='section-header'>Reset</div>", unsafe_allow_html=True)
        if st.button("🔁 Reset to Default Dataset (UniversalBank.csv)", use_container_width=True):
            st.session_state.df = load_base_data()
            st.session_state.data_source_label = "Default (UniversalBank.csv)"
            train_models.clear()
            st.success("Reset to original dataset. Navigate to any page to see updated results.")
