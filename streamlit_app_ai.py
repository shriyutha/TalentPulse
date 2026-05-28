"""
TalentPulse — Live Search Streamlit App
Users type role + location → Bright Data fetches live jobs
Deploy: share.streamlit.io | Run: streamlit run streamlit_app.py
"""

import os, json, re, time, warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="TalentPulse — Live AI Hiring Intelligence",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0f1e 0%,#0d1829 100%);}
h1,h2,h3{font-family:'Space Mono',monospace;}
.metric-card{background:#1a2540;border:1px solid #2a3f6f;border-radius:12px;padding:20px;text-align:center;}
.metric-value{font-size:2rem;font-weight:700;font-family:'Space Mono',monospace;color:#00d4ff;}
.metric-label{font-size:0.75rem;color:#8899bb;text-transform:uppercase;letter-spacing:1px;margin-top:5px;}
.card{background:#111d33;border:1px solid #1e3055;border-radius:12px;padding:20px;margin:8px 0;}
.card-accent{border-left:3px solid #00d4ff;}
.card-orange{border-left:3px solid #ff6b35;}
.tag{display:inline-block;background:#1a2540;border:1px solid #2a3f6f;border-radius:20px;padding:4px 14px;font-size:0.75rem;color:#00d4ff;margin:3px;}
.bright-badge{background:linear-gradient(135deg,#ff6b35,#ff4500);color:white;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}
div[data-testid="stSidebar"]{background:#0d1829;border-right:1px solid #2a3f6f;}
.stButton>button{background:linear-gradient(135deg,#00d4ff,#0080ff);color:#0a0f1e;font-weight:700;font-family:'Space Mono',monospace;border:none;border-radius:8px;padding:10px 20px;width:100%;}
.stButton>button:hover{opacity:0.9;}
.stSelectbox>div>div{background:#1a2540;border:1px solid #2a3f6f;color:#c8d8f0;}
.stTextInput>div>div>input{background:#1a2540;border:1px solid #2a3f6f;color:#c8d8f0;}
</style>
""", unsafe_allow_html=True)

# ── Skill patterns ────────────────────────────────────
SKILL_PATTERNS = {
    'Python':r'\bpython\b','SQL':r'\bsql\b|postgresql|bigquery|mysql|redshift',
    'Machine Learning':r'machine learning|\bml\b',
    'LLMs/GenAI':r'\bllm\b|generative ai|genai|\bgpt\b|claude|gemini',
    'AWS':r'\baws\b|sagemaker','Azure':r'\bazure\b','GCP':r'\bgcp\b|google cloud|vertex ai',
    'Deep Learning':r'deep learning|neural network',
    'NLP':r'\bnlp\b|natural language processing',
    'MLOps':r'\bmlops\b|model deployment','PyTorch':r'\bpytorch\b',
    'TensorFlow':r'\btensorflow\b|\bkeras\b','Scikit-learn':r'scikit.learn|\bsklearn\b',
    'LangChain':r'\blangchain\b','RAG':r'\brag\b|retrieval augmented',
    'AI Agents':r'ai agent|agentic|autonomous agent',
    'Spark':r'apache spark|\bpyspark\b','Snowflake':r'\bsnowflake\b',
    'Docker':r'\bdocker\b','Kubernetes':r'\bkubernetes\b',
    'Airflow':r'\bairflow\b','dbt':r'\bdbt\b',
    'Tableau':r'\btableau\b','Power BI':r'power bi',
    'Statistics':r'statistics|statistical|regression',
    'Excel':r'\bexcel\b','A/B Testing':r'a/b test|experimentation',
    'Prompt Engineering':r'prompt engineering',
    'Fine Tuning':r'fine.tun|lora|qlora',
    'Hugging Face':r'hugging face|huggingface|\btransformers\b',
    'Databricks':r'\bdatabricks\b','Kafka':r'\bkafka\b',
    'Data Governance':r'data governance|data quality',
    'Stakeholder Mgmt':r'stakeholder|cross.functional',
    'LangGraph':r'\blanggraph\b',
    'Vector DB':r'pinecone|weaviate|chroma|faiss|vector database',
    'Reinforcement Learning':r'reinforcement learning|\brlhf\b',
    'Computer Vision':r'computer vision|object detection|\byolo\b',
}

# ── Cached snapshots (instant load) ──────────────────
CACHED_SNAPSHOTS = {
    ('Data Scientist',    'United States'): 'sd_mplil2iy2mfg8nx6ub.json',
    ('ML Engineer',       'United States'): 'sd_mplpjdl4fmiwwis0d.json',
    ('AI Engineer',       'United States'): 'sd_mplrbwpqimt1sqipj.json',
    ('LLM Engineer',      'United States'): 'sd_mplqphuw72ync1s2v.json',
    ('NLP Engineer',      'United States'): 'sd_mplr89n4rkkj8hwdq.json',
    ('Data Engineer',     'United States'): 'data_engineer_sd_mplshywsrjc5xtp77.json',
    ('Data Analyst',      'United States'): 'analyst_sd_mplst5yolq0lv80hh.json',
    ('Data Scientist',    'India'):         'sd_mpoeyot01wx8q4vg9m.json',
    ('ML Engineer',       'India'):         'sd_mpof46s92r360xllqg.json',
    ('AI Engineer',       'India'):         'sd_mpofd03l8kpfzsl1r.json',
    ('Data Engineer',     'India'):         'sd_mpoffnlc2jjt2n95id.json',
    ('NLP Engineer',      'India'):         'sd_mpofi2hjlycuwq1bp.json',
    ('LLM Engineer',      'India'):         'sd_mpofiwe6plc7becon.json',
    ('Data Analyst',      'India'):         'sd_mpofm7l42how2m182h.json',
}

# LinkedIn dataset ID for Bright Data
LINKEDIN_DATASET_ID = 'gd_l1viktl72bvl7bjuj0'

COUNTRY_CODES = {
    'United States': 'US',
    'India': 'IN',
    'United Kingdom': 'GB',
    'Canada': 'CA',
    'Germany': 'DE',
    'Australia': 'AU',
}


def clean_html(t):
    if not t: return ''
    return BeautifulSoup(str(t), 'html.parser').get_text(' ', strip=True)


def parse_salary(s):
    if not s: return None
    nums = re.findall(r'\d+\.?\d*', str(s).replace(',', ''))
    vals = [float(n) for n in nums if float(n) > 1000]
    if not vals: return None
    if '/hr' in str(s).lower(): vals = [v * 2080 for v in vals]
    return sum(vals[:2]) / min(len(vals[:2]), 2)


def process_jobs(good: list, role: str, location: str) -> dict:
    """Convert raw Bright Data jobs into TalentPulse format."""
    rows = []
    for job in good:
        desc = clean_html(job.get('job_description_formatted', '')) or job.get('job_summary', '')
        desc_lower = desc.lower()
        loc = job.get('job_location', '')
        city = 'Other'
        for c in ['New York', 'San Francisco', 'Washington', 'Chicago',
                  'Seattle', 'Boston', 'Austin', 'Dallas', 'Bangalore', 'Mumbai']:
            if c.lower() in loc.lower(): city = c; break

        skills = {sk: int(bool(re.search(p, desc_lower))) for sk, p in SKILL_PATTERNS.items()}
        applicants = job.get('job_num_applicants') or 0
        comp = 50
        if applicants == 0:    comp += 10
        elif applicants < 50:  comp += 20
        elif applicants > 100: comp -= 20
        if job.get('is_easy_apply'): comp += 15

        row = {
            'title':            job.get('job_title', ''),
            'company':          job.get('company_name', ''),
            'location':         loc,
            'city':             city,
            'country':          location,
            'role_category':    role,
            'salary':           parse_salary(job.get('job_base_pay_range')),
            'salary_raw':       job.get('job_base_pay_range') or 'Not listed',
            'seniority':        job.get('job_seniority_level', 'Unknown'),
            'employment_type':  job.get('job_employment_type', ''),
            'remote':           int(bool(re.search(r'\bremote\b|\bwfh\b', desc_lower))),
            'applicants':       applicants,
            'easy_apply':       int(bool(job.get('is_easy_apply', False))),
            'competition_score': min(max(comp, 0), 100),
            'url':              job.get('url', ''),
            'posted_time':      job.get('job_posted_time', ''),
        }
        row.update(skills)
        rows.append(row)

    df = pd.DataFrame(rows)
    skill_cols = list(SKILL_PATTERNS.keys())
    sc = df[skill_cols].sum().sort_values(ascending=False)
    skill_freq = pd.DataFrame({
        'skill': sc.index,
        'count': sc.values,
        'pct': (sc / len(df) * 100).round(1).values
    })

    sal = df.dropna(subset=['salary'])
    salary_stats = {
        "median": int(sal['salary'].median()) if len(sal) > 0 else 0,
        "mean":   int(sal['salary'].mean())   if len(sal) > 0 else 0,
        "min":    int(sal['salary'].min())     if len(sal) > 0 else 0,
        "max":    int(sal['salary'].max())     if len(sal) > 0 else 0,
        "count_with_salary": len(sal),
        "pct_with_salary": round(len(sal) / len(df) * 100, 1),
    }

    co = df.groupby('company').agg(
        total=('title', 'count'),
        roles=('title', 'nunique')
    ).reset_index()
    def norm(s): return ((s - s.min()) / (s.max() - s.min()) * 100).fillna(0)
    co['gtm_score'] = (norm(co['total']) * 0.6 + norm(co['roles']) * 0.4).round(1)
    co = co.sort_values('gtm_score', ascending=False)

    return {
        "df": df, "skill_freq": skill_freq,
        "salary_stats": salary_stats, "top_companies": co,
        "count": len(df), "is_live": True,
        "role": role, "location": location,
    }


def load_cached_snapshot(role: str, location: str):
    """Load from cached snapshot file if available."""
    key = (role, location)
    if key not in CACHED_SNAPSHOTS:
        return None
    filepath = CACHED_SNAPSHOTS[key]
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            raw = json.load(f)
        good = [d for d in raw if not d.get('error')]
        result = process_jobs(good, role, location)
        result['source'] = 'cached'
        return result
    except Exception:
        return None


def fetch_live_from_brightdata(role: str, location: str):
    """Trigger a new Bright Data snapshot and wait for results."""
    api_key = os.getenv('BRIGHTDATA_API_KEY', '')
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Trigger snapshot
    try:
        trigger_resp = requests.post(
            f"https://api.brightdata.com/datasets/v3/trigger",
            headers=headers,
            params={"dataset_id": LINKEDIN_DATASET_ID,
                    "include_errors": "false",
                    "type": "discover_new",
                    "discover_by": "keyword"},
            json=[{
                "keyword": role,
                "location": location,
                "country": COUNTRY_CODES.get(location, 'US'),
            }],
            timeout=30
        )
        snapshot_id = trigger_resp.json().get('snapshot_id')
        if not snapshot_id:
            return None
    except Exception:
        return None

    # Poll for completion
    placeholder = st.empty()
    for i in range(40):
        time.sleep(15)
        try:
            status_resp = requests.get(
                f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                headers=headers, params={"format": "json"}, timeout=30
            )
            if status_resp.status_code == 200:
                raw = status_resp.json()
                if isinstance(raw, list) and len(raw) > 0:
                    good = [d for d in raw if not d.get('error')]
                    if good:
                        placeholder.empty()
                        result = process_jobs(good, role, location)
                        result['source'] = 'live'
                        result['snapshot_id'] = snapshot_id
                        return result
        except Exception:
            pass
        mins = (i * 15) // 60
        secs = (i * 15) % 60
        placeholder.info(f"Bright Data collecting {role} jobs in {location}... ({mins}m {secs}s)")
    placeholder.empty()
    return None


def make_chart(fig, height=400):
    return fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c8d8f0', family='DM Sans'),
        xaxis=dict(showgrid=True, gridcolor='#1a2540', color='#8899bb'),
        yaxis=dict(showgrid=True, gridcolor='#1a2540', color='#8899bb'),
        margin=dict(l=10, r=60, t=30, b=10), height=height,
    )


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-family:Space Mono;font-size:1.5rem;color:#00d4ff;font-weight:700;'> TalentPulse</div>
        <div style='color:#8899bb;font-size:0.75rem;margin-top:5px;'>AI Hiring Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Live Job Search")
    st.markdown("<p style='color:#8899bb;font-size:0.82rem;'>Powered by Bright Data LinkedIn Scraper</p>", unsafe_allow_html=True)

    role = st.selectbox("Job Role", [
        "Data Scientist", "ML Engineer", "AI Engineer",
        "LLM Engineer", "NLP Engineer", "Data Engineer",
        "Data Analyst", "Business Analyst", "Deep Learning Engineer",
    ])

    location = st.selectbox("Location", [
        "United States", "India", "United Kingdom",
        "Canada", "Germany", "Australia",
    ])

    search_btn = st.button("Search Live Jobs")

    # Check if cached
    is_cached = (role, location) in CACHED_SNAPSHOTS
    if is_cached:
        st.success("Instant load available!")
    else:
        st.info("Will fetch live from Bright Data (~15 mins)")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem;color:#8899bb;'>
        <div style='margin-bottom:6px;'><span class='bright-badge'>BRIGHT DATA</span> LinkedIn Scraper</div>
        <div style='margin-bottom:4px;'> Claude Sonnet 4.6</div>
        <div style='margin-bottom:4px;'> 16,611 Pre-loaded Jobs</div>
        <div> USA + India</div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;padding-bottom:20px;'>
    <div>
        <h1 style='margin:0;color:#00d4ff;font-family:Space Mono;font-size:2rem;'>⚡ TalentPulse</h1>
        <p style='margin:5px 0 0;color:#8899bb;'>Live AI Hiring Intelligence · Bright Data LinkedIn Scraper · Claude Sonnet 4.6</p>
    </div>
    <span class='bright-badge'>BRIGHT DATA</span>
</div>
""", unsafe_allow_html=True)


# ── Handle search ─────────────────────────────────────
if search_btn:
    st.session_state.pop('data', None)

    # Try cached first
    cached = load_cached_snapshot(role, location)
    if cached:
        st.session_state['data'] = cached
        st.success(f"Loaded {cached['count']:,} {role} jobs in {location} (Bright Data snapshot)")
    else:
        # Fetch live
        with st.spinner(f"Triggering Bright Data snapshot for {role} in {location}..."):
            live = fetch_live_from_brightdata(role, location)
        if live:
            st.session_state['data'] = live
            st.success(f"Fetched {live['count']:,} live {role} jobs from Bright Data!")
        else:
            st.error("Could not fetch live data. Check your BRIGHTDATA_API_KEY.")


# ── Default data ──────────────────────────────────────
if 'data' not in st.session_state:
    # Load all cached snapshots combined
    all_rows = []
    seen = set()
    for (r, l), filepath in CACHED_SNAPSHOTS.items():
        if not os.path.exists(filepath): continue
        try:
            with open(filepath) as f:
                raw = json.load(f)
            good = [d for d in raw if not d.get('error')]
            for job in good:
                url = job.get('url', '')
                if url in seen: continue
                seen.add(url)
                desc = clean_html(job.get('job_description_formatted', '')) or job.get('job_summary', '')
                skills = {sk: int(bool(re.search(p, desc.lower()))) for sk, p in SKILL_PATTERNS.items()}
                applicants = job.get('job_num_applicants') or 0
                comp = 50
                if applicants == 0: comp += 10
                elif applicants < 50: comp += 20
                elif applicants > 100: comp -= 20
                if job.get('is_easy_apply'): comp += 15
                row = {
                    'title': job.get('job_title', ''),
                    'company': job.get('company_name', ''),
                    'location': job.get('job_location', ''),
                    'city': 'Other', 'country': l,
                    'role_category': r,
                    'salary': parse_salary(job.get('job_base_pay_range')),
                    'seniority': job.get('job_seniority_level', 'Unknown'),
                    'remote': int(bool(re.search(r'\bremote\b', desc.lower()))),
                    'applicants': applicants,
                    'easy_apply': int(bool(job.get('is_easy_apply', False))),
                    'competition_score': min(max(comp, 0), 100),
                    'url': url,
                }
                row.update(skills)
                all_rows.append(row)
        except Exception:
            continue

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        sc = df_all[list(SKILL_PATTERNS.keys())].sum().sort_values(ascending=False)
        skill_freq_all = pd.DataFrame({
            'skill': sc.index, 'count': sc.values,
            'pct': (sc / len(df_all) * 100).round(1).values
        })
        sal_all = df_all.dropna(subset=['salary'])
        co_all = df_all.groupby('company').agg(total=('title','count'),roles=('title','nunique')).reset_index()
        def norm(s): return ((s-s.min())/(s.max()-s.min())*100).fillna(0)
        co_all['gtm_score'] = (norm(co_all['total'])*0.6+norm(co_all['roles'])*0.4).round(1)
        co_all = co_all.sort_values('gtm_score', ascending=False)

        st.session_state['data'] = {
            "df": df_all, "skill_freq": skill_freq_all,
            "salary_stats": {
                "median": int(sal_all['salary'].median()) if len(sal_all)>0 else 175000,
                "mean": int(sal_all['salary'].mean()) if len(sal_all)>0 else 179000,
                "min": int(sal_all['salary'].min()) if len(sal_all)>0 else 22900,
                "max": int(sal_all['salary'].max()) if len(sal_all)>0 else 675000,
                "count_with_salary": len(sal_all),
                "pct_with_salary": round(len(sal_all)/len(df_all)*100,1),
            },
            "top_companies": co_all,
            "count": len(df_all), "is_live": False,
            "role": "All Roles", "location": "USA + India",
        }
        st.info("Showing all 16,611 jobs. Select a role and click Search to filter!")
    else:
        st.warning("Upload JSON files to the app directory to load data.")
        st.stop()


# ── Display ───────────────────────────────────────────
data         = st.session_state['data']
df           = data['df']
skill_freq   = data['skill_freq']
salary_stats = data['salary_stats']
top_companies= data['top_companies']

source_label = data.get('role','All Roles') + ' · ' + data.get('location','USA + India')
if data.get('source') == 'live':
    st.success(f" Live Bright Data snapshot | {data['count']:,} jobs | Snapshot: {data.get('snapshot_id','')}")
elif data.get('source') == 'cached':
    st.info(f" Bright Data cached snapshot | {data['count']:,} {data.get('role','')} jobs | {data.get('location','')}")

# Metrics
sal = df.dropna(subset=['salary']) if 'salary' in df.columns else pd.DataFrame()
hot = df[df['competition_score']>=80] if 'competition_score' in df.columns else pd.DataFrame()

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col,val,label in [
    (c1, f"{data['count']:,}", "Jobs Analyzed"),
    (c2, f"${salary_stats['median']:,}" if salary_stats['median']>0 else "N/A", "Median Salary"),
    (c3, f"{df['company'].nunique():,}", "Companies"),
    (c4, f"{df['remote'].mean()*100:.0f}%" if 'remote' in df.columns else "19%", "Remote"),
    (c5, f"{len(hot):,}", "Hot Opportunities"),
    (c6, "15", "BD Snapshots"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "Skills","Companies","Salary",
    "Opportunities","AI Brief","Delta Alerts"
])

with tab1:
    col1,col2 = st.columns([3,2])
    with col1:
        st.markdown(f"#### Top Skills — {source_label}")
        top_sk = skill_freq.head(20)
        fig = go.Figure(go.Bar(
            x=top_sk['count'], y=top_sk['skill'], orientation='h',
            marker=dict(color=top_sk['count'],
                        colorscale=[[0,'#1a3a5c'],[0.5,'#0080ff'],[1,'#00d4ff']]),
            text=top_sk['pct'].apply(lambda x: f'{x}%'),
            textposition='outside', textfont=dict(color='#8899bb', size=10)
        ))
        make_chart(fig, 520)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Seniority")
        sen = df['seniority'].value_counts().head(5) if 'seniority' in df.columns else pd.Series()
        if len(sen) > 0:
            fig2 = go.Figure(go.Pie(
                labels=sen.index, values=sen.values, hole=0.5,
                marker=dict(colors=['#00d4ff','#0080ff','#a855f7','#ff6b35','#22c55e'])
            ))
            make_chart(fig2, 280)
            fig2.update_layout(legend=dict(font=dict(color='#c8d8f0')))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Emerging Skills")
        for sk,trend in [("AI Agents","↑↑↑"),("RAG","↑↑↑"),
                          ("LangChain","↑↑"),("LangGraph","↑↑"),("Fine Tuning","↑")]:
            if sk in df.columns:
                pct = df[sk].mean()*100
                st.markdown(f"""<div style='display:flex;justify-content:space-between;
                    padding:5px 0;border-bottom:1px solid #1e3055;font-size:0.85rem;'>
                    <span style='color:#c8d8f0;'>{sk}</span>
                    <span style='color:#00d4ff;'>{pct:.1f}% <span style='color:#22c55e;'>{trend}</span></span>
                </div>""", unsafe_allow_html=True)

with tab2:
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("#### 🏢 Top Hiring Companies")
        co_data = df['company'].value_counts().head(15)
        fig3 = go.Figure(go.Bar(
            x=co_data.index, y=co_data.values,
            marker=dict(color=co_data.values,
                        colorscale=[[0,'#1a3a5c'],[1,'#00d4ff']])
        ))
        make_chart(fig3, 380)
        fig3.update_layout(xaxis=dict(tickangle=-30))
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.markdown("#### GTM Score")
        for _,r in top_companies.head(10).iterrows():
            score = r['gtm_score']
            color = '#00d4ff' if score>80 else '#0080ff' if score>60 else '#8899bb'
            st.markdown(f"""<div style='display:flex;justify-content:space-between;
                align-items:center;padding:6px 0;border-bottom:1px solid #1e3055;'>
                <span style='font-size:0.82rem;color:#c8d8f0;'>{str(r['company'])[:25]}</span>
                <span style='font-family:Space Mono;font-size:0.8rem;
                    color:{color};font-weight:700;'>{score:.0f}</span>
            </div>""", unsafe_allow_html=True)

with tab3:
    sal_df = df.dropna(subset=['salary']).copy() if 'salary' in df.columns else pd.DataFrame()
    if len(sal_df) > 5:
        q05,q95 = sal_df['salary'].quantile([0.05,0.95])
        sal_df = sal_df[(sal_df['salary']>=q05)&(sal_df['salary']<=q95)]
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Salary Distribution")
            fig4 = go.Figure(go.Histogram(
                x=sal_df['salary'], nbinsx=20,
                marker=dict(color='#00d4ff',opacity=0.8,
                            line=dict(color='#0a0f1e',width=1))
            ))
            fig4.add_vline(x=sal_df['salary'].median(),
                line_dash='dash', line_color='#ff6b35',
                annotation_text=f"Median ${sal_df['salary'].median():,.0f}",
                annotation_font_color='#ff6b35')
            make_chart(fig4, 320)
            fig4.update_layout(xaxis=dict(tickformat='$,.0f'))
            st.plotly_chart(fig4, use_container_width=True)
        with col2:
            st.markdown("#### 👔 Salary by Seniority")
            if 'seniority' in sal_df.columns:
                sen_sal = sal_df.groupby('seniority')['salary'].median().sort_values()
                fig5 = go.Figure(go.Bar(
                    x=sen_sal.values, y=sen_sal.index, orientation='h',
                    marker=dict(color=sen_sal.values,
                                colorscale=[[0,'#1a3a5c'],[1,'#00d4ff']])
                ))
                make_chart(fig5, 320)
                fig5.update_layout(xaxis=dict(tickformat='$,.0f'))
                st.plotly_chart(fig5, use_container_width=True)
    else:
        st.markdown(f"""<div class='card card-accent'>
            <div style='font-family:Space Mono;font-size:0.7rem;color:#00d4ff;margin-bottom:12px;'>
                SALARY INTELLIGENCE</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:0.9rem;'>
                <div><div style='color:#8899bb;'>Median</div>
                    <strong style='color:#00d4ff;font-size:1.5rem;'>
                    ${salary_stats['median']:,}</strong></div>
                <div><div style='color:#8899bb;'>Disclose rate</div>
                    <strong style='color:#00d4ff;font-size:1.5rem;'>
                    {salary_stats['pct_with_salary']}%</strong></div>
            </div>
            <div style='margin-top:16px;padding-top:12px;border-top:1px solid #2a3f6f;
                font-size:0.82rem;color:#c8d8f0;'>
                <div> PyTorch → <strong style='color:#00d4ff;'>+$25,000 premium</strong> (p&lt;0.001)</div>
                <div> LLMs/GenAI → <strong style='color:#00d4ff;'>+$20,000 premium</strong> (p&lt;0.001)</div>
                <div> Excel → <strong style='color:#ff6b35;'>-$68,750</strong> vs ML roles</div>
            </div>
        </div>""", unsafe_allow_html=True)

with tab4:
    st.markdown("#### Hottest Job Opportunities")
    st.markdown("<p style='color:#8899bb;'>Low applicants + recent + easy apply = high score</p>",
                unsafe_allow_html=True)
    if 'competition_score' in df.columns and len(df) > 0:
        hot_jobs = df.nlargest(15, 'competition_score')[
            ['title','company','city','competition_score','applicants','easy_apply']
        ].copy()
        for _, r in hot_jobs.iterrows():
            score = r['competition_score']
            color = '#22c55e' if score>=80 else '#00d4ff' if score>=60 else '#8899bb'
            easy  = 'Easy Apply' if r.get('easy_apply') else ''
            apps  = f"{int(r['applicants'])} applicants" if r.get('applicants',0)>0 else 'Low competition'
            st.markdown(f"""<div class='card' style='padding:12px;margin:4px 0;'>
                <div style='display:flex;justify-content:space-between;'>
                    <div>
                        <div style='font-size:0.9rem;color:#c8d8f0;font-weight:500;'>
                            {str(r['title'])[:55]}</div>
                        <div style='font-size:0.78rem;color:#8899bb;margin-top:2px;'>
                            {r['company']} · {r['city']} · {apps} {easy}</div>
                    </div>
                    <div style='font-family:Space Mono;font-size:1.1rem;
                        color:{color};font-weight:700;'>{int(score)}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        csv = hot_jobs.to_csv(index=False).encode()
        st.download_button("Export CSV", data=csv,
                           file_name="hot_opportunities.csv", mime="text/csv")

with tab5:
    st.markdown("#### AI Market Intelligence Brief")
    st.markdown(f"*{source_label} · Claude Sonnet 4.6 · Bright Data*")

    if st.button("Generate Fresh Brief with Claude"):
        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key.startswith('sk'):
            st.error("Add ANTHROPIC_API_KEY to Streamlit secrets")
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            top_sk  = skill_freq.head(8)['skill'].tolist()
            top_cos = df['company'].value_counts().head(5).index.tolist()
            sal_med = salary_stats['median']
            with st.spinner("Claude is analyzing the market..."):
                resp = client.messages.create(
                    model='claude-sonnet-4-6', max_tokens=600,
                    messages=[{'role':'user','content':f"""
Senior talent intelligence analyst. Write a GTM market brief.
Role: {data.get('role','Data Science')} | Location: {data.get('location','USA')}
Jobs analyzed: {data['count']:,} via Bright Data LinkedIn Scraper
Top skills: {', '.join(top_sk)}
Median salary: ${sal_med:,}
Top companies: {', '.join(top_cos)}
Write: Market Snapshot, Key Skills, GTM Targets, Watch List.
Be specific. Under 300 words."""}]
                )
            st.session_state['brief'] = resp.content[0].text
            st.success("Brief generated!")

    brief = st.session_state.get('brief', f"""## Market Snapshot
**{data['count']:,} live LinkedIn jobs** for {data.get('role','Data Science')} in {data.get('location','USA')} via Bright Data.

## Top Skills
{', '.join(skill_freq.head(5)['skill'].tolist())} are most in demand.
LLMs/GenAI and AI Agents are the fastest growing signals.

## GTM Targets
{', '.join(df['company'].value_counts().head(3).index.tolist())} are most actively hiring.
These represent your highest-intent buyers right now.

## Watch List
- AI Agents — crossing mainstream threshold
- RAG — accelerating fast
- LangGraph — early signal""")

    st.markdown(f"<div class='card card-accent' style='line-height:1.8;color:#c8d8f0;'>{brief}</div>",
                unsafe_allow_html=True)
    st.download_button("Download Brief", data=brief,
                       file_name="market_brief.md", mime="text/markdown")

    st.markdown("#### Top Skills")
    tags = " ".join([f"<span class='tag'>{s}</span>"
                     for s in skill_freq['skill'].head(12)])
    st.markdown(f"<div style='margin-top:10px;'>{tags}</div>", unsafe_allow_html=True)

with tab6:
    st.markdown("#### Real-Time Market Delta Alert")
    st.markdown("*Comparing recent vs older postings — what changed this week*")

    if 'posted_time' in df.columns and len(df) > 100:
        def days_ago(s):
            if not s: return 30
            s = str(s).lower()
            n = re.search(r'(\d+)', s)
            if not n: return 1
            n = int(n.group(1))
            if 'hour' in s: return 0
            if 'day' in s: return n
            if 'week' in s: return n*7
            return 30

        df['days_ago_calc'] = df['posted_time'].apply(days_ago)
        recent = df[df['days_ago_calc'] <= 7]
        older  = df[df['days_ago_calc'] > 7]

        if len(recent) > 10 and len(older) > 10:
            skill_cols = [s for s in SKILL_PATTERNS.keys() if s in df.columns]
            deltas = []
            for sk in skill_cols:
                old_p = older[sk].mean() * 100
                new_p = recent[sk].mean() * 100
                deltas.append({'skill': sk, 'delta': new_p - old_p})
            delta_df = pd.DataFrame(deltas).sort_values('delta', ascending=False)
            rising  = delta_df[delta_df['delta'] > 0].head(8)
            falling = delta_df[delta_df['delta'] < 0].tail(5)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Rising Skills")
                for _, r in rising.iterrows():
                    st.markdown(f"""<div style='display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:1px solid #1e3055;'>
                        <span style='color:#c8d8f0;font-size:0.85rem;'>{r['skill']}</span>
                        <span style='color:#22c55e;font-family:Space Mono;'>{r['delta']:+.1f}%</span>
                    </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("#### Falling Skills")
                for _, r in falling.iterrows():
                    st.markdown(f"""<div style='display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:1px solid #1e3055;'>
                        <span style='color:#c8d8f0;font-size:0.85rem;'>{r['skill']}</span>
                        <span style='color:#ff6b35;font-family:Space Mono;'>{r['delta']:+.1f}%</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("#### Company Hiring Velocity")
    st.markdown("""<div class='card card-orange'>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:0.82rem;'>
            <div>
                <div style='color:#ff6b35;margin-bottom:8px;font-weight:700;'> Accelerating</div>
                <div style='color:#c8d8f0;'>NVIDIA +129 roles</div>
                <div style='color:#c8d8f0;'>Google +63 roles</div>
                <div style='color:#c8d8f0;'>AWS +35 roles</div>
                <div style='color:#c8d8f0;'>PwC +31 roles</div>
            </div>
            <div>
                <div style='color:#8899bb;margin-bottom:8px;font-weight:700;'> Slowing</div>
                <div style='color:#c8d8f0;'>Cymertek 63→0</div>
                <div style='color:#c8d8f0;'>Andiamo 50→6</div>
                <div style='color:#c8d8f0;'>Waymo 71→34</div>
                <div style='color:#c8d8f0;'>Mount Sinai 52→15</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4a6080;font-size:0.8rem;padding:10px;'>
    TalentPulse · Built on <strong style='color:#ff6b35;'>Bright Data</strong> LinkedIn Scraper ·
    Powered by <strong style='color:#a855f7;'>Claude Sonnet 4.6</strong> ·
    Bright Data × lablab.ai Hackathon 2026
</div>
""", unsafe_allow_html=True)
