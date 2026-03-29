"""Bloomberg Terminal 风格自定义样式。"""

_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ======== Base Typography ======== */
.stApp, .stApp > div, .stApp > div > div {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    color: #E8E6E3 !important;
}

h1, h2, h3, h4, h5, h6, .stApp h1, .stApp h2, .stApp h3 {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    font-weight: 600 !important;
    color: #E8E6E3 !important;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ======== Layout ======== */
[data-testid="stMain"] {
    background-color: #0A0E27 !important;
    padding-top: 2.5rem !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ======== Sidebar ======== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1421 0%, #091018 100%) !important;
    border-right: 2px solid #C9A84C !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}

/* Sidebar section headers */
[data-testid="stSidebar"] h3 {
    color: #C9A84C !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.3rem !important;
    border-bottom: 1px solid rgba(201,168,76,0.2) !important;
    padding-bottom: 0.25rem !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #C9A84C !important;
    color: #C9A84C !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border-radius: 4px !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(201,168,76,0.12) !important;
    border-color: #D4AF37 !important;
    color: #D4AF37 !important;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.8rem !important;
    color: #9CA3AF !important;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span[data-baseweb="radio-dot"] {
    background-color: #C9A84C !important;
}

/* Sidebar sliders */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] {
    background-color: #1E293B !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-baseweb="slider-fill"] {
    background: linear-gradient(90deg, #8B7D3C, #C9A84C) !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-baseweb="slider-thumb"] {
    background-color: #C9A84C !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background-color: #111832 !important;
    border-color: #1E293B !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
    color: #E8E6E3 !important;
}

/* Sidebar text input */
[data-testid="stSidebar"] .stTextInput [data-baseweb="base-input"] {
    background-color: #111832 !important;
    border-color: #1E293B !important;
    color: #E8E6E3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] .stTextInput [data-baseweb="base-input"]:focus {
    border-color: #C9A84C !important;
}

/* Sidebar alerts */
[data-testid="stSidebar"] .stAlert {
    background-color: rgba(17,24,42,0.8) !important;
    border: 1px solid #1E293B !important;
    font-size: 0.8rem !important;
}

/* Sidebar slider label */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p {
    font-size: 0.75rem !important;
    color: #9CA3AF !important;
}

/* ======== Tabs ======== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid rgba(201,168,76,0.25) !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #9CA3AF !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #E8E6E3 !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #C9A84C !important;
    border-bottom: 2px solid #C9A84C !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #C9A84C !important;
}

[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ======== Chat Messages ======== */
[data-testid="stChatMessage"] {
    border-radius: 4px !important;
    padding: 0.8rem 1rem !important;
    border: 1px solid #1E293B !important;
    margin-bottom: 0.5rem !important;
    background-color: #0D1421 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #111832 !important;
    border-left: 3px solid #C9A84C !important;
}

/* Chat avatars */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    background-color: transparent !important;
    border: 1px solid #1E293B !important;
    border-radius: 4px !important;
}

/* Chat input */
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background-color: #111832 !important;
    border: 1px solid #1E293B !important;
    color: #E8E6E3 !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"]:focus {
    border-color: #C9A84C !important;
}

[data-testid="stChatInput"] [data-baseweb="base-input"]::placeholder {
    color: #4B5563 !important;
}

/* ======== Custom HTML Components (injected via st.markdown) ======== */
.main-header {
    text-align: center;
    padding: 0.4rem 0 0.3rem 0 !important;
    margin: 0 !important;
    border-bottom: 1px solid rgba(201,168,76,0.25);
}

.main-header h1 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #C9A84C !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.4 !important;
}

.main-header .subtitle {
    font-size: 0.7rem !important;
    color: #4B5563 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin: 0.1rem 0 0 0 !important;
    line-height: 1.3 !important;
}

.branding-footer {
    margin-top: 1.5rem;
    padding: 0.5rem 0;
    border-top: 1px solid rgba(201,168,76,0.2);
    text-align: center;
}

.branding-footer .brand-name {
    font-size: 0.7rem;
    color: #C9A84C;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.branding-footer .brand-version {
    font-size: 0.6rem;
    color: #3B4559;
    letter-spacing: 1px;
    margin-top: 0.15rem;
}

.sidebar-kb-stat {
    background: rgba(17,24,42,0.6);
    border: 1px solid #1E293B;
    border-radius: 4px;
    padding: 0.5rem 0.7rem;
    margin-top: 0.2rem;
}

.sidebar-kb-stat .stat-label {
    font-size: 0.65rem;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sidebar-kb-stat .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #C9A84C;
    font-weight: 600;
    margin-top: 0.1rem;
}

/* ======== Expanders ======== */
[data-testid="stExpander"] {
    border: 1px solid #1E293B !important;
    border-radius: 4px !important;
    background-color: #0D1421 !important;
}

[data-testid="stExpander"] details {
    background-color: #0A0E27 !important;
}

[data-testid="stExpander"] details summary {
    color: #C9A84C !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.8rem !important;
}

/* ======== Buttons (Main Area) ======== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #C9A84C, #D4AF37) !important;
    color: #0A0E27 !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.2rem !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #D4AF37, #E0BC5A) !important;
    box-shadow: 0 0 16px rgba(201,168,76,0.25) !important;
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #C9A84C !important;
    color: #C9A84C !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border-radius: 4px !important;
    padding: 0.4rem 1rem !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(201,168,76,0.08) !important;
}

/* ======== File Uploader ======== */
[data-testid="stFileUploader"] section {
    background-color: #0D1421 !important;
    border: 2px dashed rgba(201,168,76,0.35) !important;
    border-radius: 6px !important;
    padding: 1.5rem !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(201,168,76,0.6) !important;
}

[data-testid="stFileUploader"] section [data-baseweb="form-control"] label {
    color: #9CA3AF !important;
    font-size: 0.85rem !important;
}

/* ======== DataFrames / Tables ======== */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] table {
    border-collapse: collapse !important;
}

[data-testid="stDataFrame"] thead th {
    background-color: #0D1421 !important;
    color: #C9A84C !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 0.5rem 0.8rem !important;
    border-bottom: 2px solid rgba(201,168,76,0.3) !important;
}

[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: rgba(13,20,33,0.5) !important;
}

[data-testid="stDataFrame"] tbody tr:hover {
    background-color: rgba(201,168,76,0.06) !important;
}

[data-testid="stDataFrame"] tbody td {
    color: #E8E6E3 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.8rem !important;
}

/* ======== Progress Bars ======== */
[data-testid="stProgress"] [data-baseweb="progress-bar"] {
    background-color: #1E293B !important;
}

[data-testid="stProgress"] [data-baseweb="progress-bar"] [data-baseweb="progress-bar-inner"] {
    background: linear-gradient(90deg, #8B7D3C, #C9A84C, #D4AF37) !important;
}

/* ======== Alerts ======== */
.stAlert {
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    border: 1px solid #1E293B !important;
}

.stAlert[data-testid="stAlert"] {
    background-color: rgba(17,24,42,0.9) !important;
}

.stAlert [data-baseweb="notification"] {
    background-color: transparent !important;
}

/* ======== Checkbox ======== */
.stCheckbox label {
    color: #9CA3AF !important;
    font-size: 0.85rem !important;
}

/* ======== Caption ======== */
[data-testid="stCaption"] {
    color: #4B5563 !important;
    font-size: 0.78rem !important;
}

/* ======== Subheaders (Main Area) ======== */
[data-testid="stMain"] h3 {
    color: #C9A84C !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

/* ======== Markdown Content ======== */
.stMarkdown strong {
    color: #C9A84C !important;
    font-weight: 600 !important;
}

.stMarkdown code {
    background-color: rgba(17,24,42,0.8) !important;
    border: 1px solid #1E293B !important;
    border-radius: 3px !important;
    padding: 0.1rem 0.3rem !important;
    font-size: 0.8rem !important;
    color: #D4AF37 !important;
}

/* ======== Slider Labels (Main Area) ======== */
.stSlider label, .stSlider [data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    color: #9CA3AF !important;
}

/* ======== Scrollbars ======== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E27; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A5C; }
"""
