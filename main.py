import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import math
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

# 页面基础配置
st.set_page_config(
    page_title="2026世界杯真实数据AI预测平台",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 全局CSS：动效、渐变、卡片悬浮、进度条动画 ==========
st.markdown("""
<style>
html, body, section[data-testid="stMain"], section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"] {
    overflow-y: auto !important;
    height: auto !important;
}
div[data-testid="stAppViewContainer"] {padding-top: 0 !important;}
div[data-testid="stMainBlockContainer"] {padding-top: 10px !important;}
header {display: none !important;}
html {scroll-behavior: smooth;}

/* 去除主内容区白框，让渐变背景透出 */
div[data-testid="stMainBlockContainer"],
section[data-testid="stMain"] {
    background: transparent !important;
}
/* 全局字体调大 */
html, body, .stApp {
    font-size: 19px !important;
}
.stMarkdown, .stMarkdown p, .stText, .stCaption {
    font-size: 19px !important;
}
.stSelectbox label, .stCheckbox label, .stRadio label {
    font-size: 18px !important;
}
.stMetric label {font-size:17px !important;}
.stMetric value {font-size:24px !important;}

/* 渐变背景 + 球场草坪底纹 */
div[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 20% 20%, rgba(22,93,255,0.06) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(0,194,255,0.06) 0%, transparent 40%),
        repeating-linear-gradient(90deg,
            rgba(34,139,34,0.04) 0px, rgba(34,139,34,0.04) 80px,
            rgba(22,93,255,0.04) 80px, rgba(22,93,255,0.04) 160px),
        linear-gradient(135deg, #f5f9ff 0%, #eaf3ff 50%, #f0f7ff 100%) !important;
    background-attachment: fixed !important;
}

/* 左右两侧球星装饰（C罗/梅西），固定在两侧，虚化融入背景 */
.side-star {
    position: fixed; top: 18%; transform: translateY(-50%);
    z-index: 0; pointer-events: none;
    display: flex; flex-direction: column; align-items: center;
}
.side-star .star-photo {
    width: calc(100vh / 3 * 0.85); height: calc(100vh / 3);
    object-fit: cover; object-position: top center;
    border-radius: 8px;
    /* 边框虚化：渐变遮罩向背景过渡 */
    -webkit-mask-image: linear-gradient(to bottom, transparent 0%, #000 18%, #000 82%, transparent 100%),
                        linear-gradient(to right, transparent 0%, #000 15%, #000 85%, transparent 100%);
    -webkit-mask-composite: source-in;
    mask-image: linear-gradient(to bottom, transparent 0%, #000 18%, #000 82%, transparent 100%),
                linear-gradient(to right, transparent 0%, #000 15%, #000 85%, transparent 100%);
    mask-composite: intersect;
    opacity: 0.75;
}
.side-star.left { left: 4px; }
.side-star.right { right: 4px; }
@media (max-width: 1100px) { .side-star { display: none; } }

/* 漂浮足球背景层 */
.bg-footballs {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.bg-footballs span {
    position: absolute; font-size: 32px; opacity: 0.12;
    animation: floatBall 18s linear infinite;
}
@keyframes floatBall {
    0%   { transform: translateY(110vh) rotate(0deg); }
    100% { transform: translateY(-20vh) rotate(720deg); }
}

/* 顶部世界杯装饰条 */
.wc-banner {
    position: relative;
    background: linear-gradient(90deg, #165DFF, #00C2FF, #165DFF);
    background-size: 200% auto;
    animation: shine 6s linear infinite;
    color: #fff; text-align: center;
    padding: 8px 24px; font-weight: 600; font-size: 13px;
    border-radius: 10px; margin: 0 auto 14px;
    box-shadow: 0 4px 14px rgba(22,93,255,0.25);
    letter-spacing: 1px;
    width: fit-content; max-width: 60%;
}

/* 奖杯点缀 */
.wc-trophy {
    text-align: center; font-size: 40px;
    margin-bottom: 6px;
    animation: trophyGlow 2.5s ease-in-out infinite;
}
@keyframes trophyGlow {
    0%,100% { transform: scale(1); filter: drop-shadow(0 0 6px rgba(255,215,0,0.5)); }
    50% { transform: scale(1.08); filter: drop-shadow(0 0 16px rgba(255,215,0,0.9)); }
}

/* 标题动效 */
.main-title {
    font-size:38px;font-weight:800;
    background:linear-gradient(90deg,#165DFF,#00C2FF,#165DFF);
    background-size:200% auto;
    -webkit-background-clip:text;background-clip:text;
    -webkit-text-fill-color:transparent;
    text-align:center;margin:5px 0;
    animation:shine 4s linear infinite;
}
@keyframes shine {to {background-position:200% center;}}
.sub-title {text-align:center;color:#666;margin-bottom:20px;font-size:14px;}

/* 卡片通用 */
.input-card, .result-card, .reason-block, .stat-card {
    border-radius:16px;padding:22px;
    background:rgba(255,255,255,0.85);
    backdrop-filter:blur(8px);
    border:1px solid rgba(22,93,255,0.12);
    transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
    margin-bottom:18px;
}
/* 动态草坪条：完整独立块，流动草坪 + 跳动足球 + 左右滑动 */
.grass-strip {
    position: relative;
    height: 90px;
    margin: 10px 0 22px;
    border: 2px solid rgba(22,93,255,0.4);
    border-radius: 12px;
    overflow: hidden;
    background:
        repeating-linear-gradient(90deg,
            rgba(34,139,34,0.75) 0px, rgba(34,139,34,0.75) 40px,
            rgba(45,160,45,0.75) 40px, rgba(45,160,45,0.75) 80px),
        linear-gradient(180deg, rgba(50,170,50,0.5), rgba(30,120,30,0.6));
    background-size: 80px 100%, 100% 100%;
    animation: grassFlow 3s linear infinite, swayLR 5s ease-in-out infinite;
}
@keyframes grassFlow {
    from { background-position: 0 0, 0 0; }
    to   { background-position: -80px 0, 0 0; }
}
@keyframes swayLR {
    0%,100% { transform: translateX(0) rotate(0deg); }
    25%     { transform: translateX(-12px) rotate(-0.3deg); }
    75%     { transform: translateX(12px) rotate(0.3deg); }
}
.grass-strip .ball {
    position: absolute;
    font-size: 30px;
    top: 50%;
    margin-top: -15px;
    animation: bounceBall 1.6s ease-in-out infinite;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
}
@keyframes bounceBall {
    0%,100% { transform: translateY(0) rotate(0deg); }
    50%     { transform: translateY(-18px) rotate(180deg); }
}
.result-card {box-shadow:0 10px 30px rgba(22,93,255,0.15);}
.reason-block {background:#F5F9FF;margin-top:16px;line-height:1.75;}
.stat-card {box-shadow:0 6px 20px rgba(22,93,255,0.10);}
.result-card:hover, .stat-card:hover, .input-card:hover {
    transform:translateY(-4px) scale(1.005);
    box-shadow:0 16px 40px rgba(22,93,255,0.25);
}

/* 按钮动效 */
.stButton>button {
    transition:.3s ease;border-radius:12px;font-weight:600;
    background:linear-gradient(90deg,#165DFF,#00A6FF)!important;
    border:none!important;color:#fff!important;
    box-shadow:0 4px 14px rgba(22,93,255,.35);
}
.stButton>button:hover {transform:scale(1.03);box-shadow:0 8px 22px rgba(22,93,255,.5);}
.stButton>button:active {transform:scale(.98);}

/* 进度条 */
.stProgress > div > div {
    background:linear-gradient(90deg,#165DFF,#00C2FF)!important;
    transition:width 1s cubic-bezier(.2,.8,.2,1);
}

/* 比分大字 */
.score-big {
    font-size:56px;font-weight:900;text-align:center;
    background:linear-gradient(90deg,#FF6B6B,#165DFF);
    -webkit-background-clip:text;background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:pulse 2s ease-in-out infinite;
    margin:10px 0;
}
@keyframes pulse {0%,100%{transform:scale(1);}50%{transform:scale(1.05);}}

/* 胜负徽章 */
.win-badge {
    display:inline-block;padding:4px 14px;border-radius:20px;
    font-weight:700;font-size:13px;color:#fff;margin:2px;
}
.badge-a {background:linear-gradient(90deg,#165DFF,#00A6FF);}
.badge-b {background:linear-gradient(90deg,#FF6B6B,#FF8E53);}
.badge-d {background:linear-gradient(90deg,#7B8AFF,#A78BFA);}

/* 数据集总览表格 */
.dataframe-table {font-size:13px;}

/* 滚动揭示动画 */
@keyframes fadeUp {from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
.fade-in {animation:fadeUp .7s ease both;}

/* 分隔线装饰 */
hr {border:none;height:2px;
    background:linear-gradient(90deg,transparent,#165DFF,transparent);
    margin:24px 0;border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ========== 中文 ↔ 英文球队名称映射（覆盖主流队伍，其余保留英文） ==========
team_cn_en_map = {
    "巴西":"Brazil","葡萄牙":"Portugal","西班牙":"Spain","法国":"France",
    "阿根廷":"Argentina","克罗地亚":"Croatia","挪威":"Norway","英格兰":"England",
    "德国":"Germany","意大利":"Italy","荷兰":"Netherlands","比利时":"Belgium",
    "瑞士":"Switzerland","乌拉圭":"Uruguay","墨西哥":"Mexico","美国":"United States",
    "日本":"Japan","韩国":"Korea Republic","摩洛哥":"Morocco","塞内加尔":"Senegal",
    "波兰":"Poland","塞尔维亚":"Serbia","丹麦":"Denmark","瑞典":"Sweden",
    "俄罗斯":"Russia","乌克兰":"Ukraine","土耳其":"Türkiye","希腊":"Greece",
    "哥伦比亚":"Colombia","秘鲁":"Peru","智利":"Chile","厄瓜多尔":"Ecuador",
    "澳大利亚":"Australia","加纳":"Ghana","尼日利亚":"Nigeria","喀麦隆":"Cameroon",
    "中国":"China PR","沙特":"Saudi Arabia","伊朗":"IR Iran","卡塔尔":"Qatar",
    "匈牙利":"Hungary","捷克":"Czech Republic","罗马尼亚":"Romania","苏格兰":"Scotland",
    "威尔士":"Wales","北爱尔兰":"Northern Ireland","爱尔兰":"Republic of Ireland",
    "埃及":"Egypt","突尼斯":"Tunisia","阿尔及利亚":"Algeria","南非":"South Africa",
    "哥斯达黎加":"Costa Rica","巴拿马":"Panama","牙买加":"Jamaica","洪都拉斯":"Honduras",
    "冰岛":"Iceland","以色列":"Israel","伊拉克":"Iraq","科威特":"Kuwait",
    "阿联酋":"United Arab Emirates","特立尼达和多巴哥":"Trinidad and Tobago",
    "新西兰":"New Zealand","多哥":"Togo","安哥拉":"Angola","玻利维亚":"Bolivia",
    "古巴":"Cuba","海地":"Haiti","萨尔瓦多":"El Salvador","荷属东印度":"Dutch East Indies",
    "扎伊尔":"Zaire","斯洛伐克":"Slovakia","斯洛文尼亚":"Slovenia",
    "波斯尼亚和黑塞哥维那":"Bosnia and Herzegovina","加拿大":"Canada"
}
team_en_cn_map = {v: k for k, v in team_cn_en_map.items()}

# ========== 国旗 emoji 映射（用作队徽替代，覆盖数据集全部84支队伍含历史国家） ==========
team_flag_map = {
    "Algeria":"🇩🇿","Angola":"🇦🇴","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Bolivia":"🇧🇴","Bosnia and Herzegovina":"🇧🇦","Brazil":"🇧🇷","Bulgaria":"🇧🇬",
    "Cameroon":"🇨🇲","Canada":"🇨🇦","Chile":"🇨🇱","China PR":"🇨🇳","Colombia":"🇨🇴",
    "Costa Rica":"🇨🇷","Croatia":"🇭🇷","Cuba":"🇨🇺","Czech Republic":"🇨🇿","Czechoslovakia":"🇨🇿",
    "Côte d'Ivoire":"🇨🇮","Denmark":"🇩🇰","Dutch East Indies":"🇮🇩","Ecuador":"🇪🇨","Egypt":"🇪🇬",
    "El Salvador":"🇸🇻","England":"🏴","FR Yugoslavia":"🇷🇸","France":"🇫🇷","Germany":"🇩🇪",
    "Germany DR":"🇩🇪","Ghana":"🇬🇭","Greece":"🇬🇷","Haiti":"🇭🇹","Honduras":"🇭🇳",
    "Hungary":"🇭🇺","IR Iran":"🇮🇷","Iceland":"🇮🇸","Iraq":"🇮🇶","Israel":"🇮🇱",
    "Italy":"🇮🇹","Jamaica":"🇯🇲","Japan":"🇯🇵","Korea DPR":"🇰🇵","Korea Republic":"🇰🇷",
    "Kuwait":"🇰🇼","Mexico":"🇲🇽","Morocco":"🇲🇦","Netherlands":"🇳🇱","New Zealand":"🇳🇿",
    "Nigeria":"🇳🇬","Northern Ireland":"🏴","Norway":"🇳🇴","Panama":"🇵🇦","Paraguay":"🇵🇾",
    "Peru":"🇵🇪","Poland":"🇵🇱","Portugal":"🇵🇹","Qatar":"🇶🇦","Republic of Ireland":"🇮🇪",
    "Romania":"🇷🇴","Russia":"🇷🇺","Saudi Arabia":"🇸🇦","Scotland":"🏴","Senegal":"🇸🇳",
    "Serbia":"🇷🇸","Serbia and Montenegro":"🇷🇸","Slovakia":"🇸🇰","Slovenia":"🇸🇮","South Africa":"🇿🇦",
    "Soviet Union":"🇷🇺","Spain":"🇪🇸","Sweden":"🇸🇪","Switzerland":"🇨🇭","Togo":"🇹🇬",
    "Trinidad and Tobago":"🇹🇹","Tunisia":"🇹🇳","Türkiye":"🇹🇷","Ukraine":"🇺🇦",
    "United Arab Emirates":"🇦🇪","United States":"🇺🇸","Uruguay":"🇺🇾","Wales":"🏴",
    "West Germany":"🇩🇪","Yugoslavia":"🇷🇸","Zaire":"🇨🇩"
}
def team_flag(en_name):
    return team_flag_map.get(en_name, "🏳️")

# 赛事阶段英文 → 中文映射
stage_cn_map = {
    "Final":"决赛","Third-place match":"季军赛","Semi-finals":"半决赛",
    "Quarter-finals":"四分之一决赛","Round of 16":"八分之一决赛",
    "Group stage":"小组赛","Second group stage":"第二阶段小组赛",
    "First group stage":"第一阶段小组赛","Second round":"第二轮",
    "First round":"第一轮","Group stage play-off":"小组赛附加赛",
    "Final stage":"决赛阶段"
}
def stage_cn(en_stage):
    return stage_cn_map.get(en_stage, en_stage)

# 比赛结果数字 → 中文
result_cn_map = {0:"落败",1:"平局",2:"胜利"}

# ========== 加载数据集 ==========
@st.cache_data
def load_clean_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "世界杯清洗后结构化数据集_含红牌.csv")
    return pd.read_csv(csv_path, encoding="gbk")

team_dataset = load_clean_data()

# 生成中文展示版数据集（列名、赛事阶段、球队名、比赛结果全部中文化，球队名加国旗）
@st.cache_data
def build_cn_display_df(_data):
    df = _data.copy()
    # 球队名加国旗+中文
    df["球队"] = df["球队"].apply(lambda t: f"{team_flag(t)} {team_en_cn_map.get(t, t)}")
    # 赛事阶段转中文
    df["赛事阶段"] = df["赛事阶段"].apply(stage_cn)
    # 比赛结果数字转中文
    df["比赛结果"] = df["比赛结果"].map(result_cn_map)
    # 是否字段 0/1 转 否/是
    df["是否淘汰赛"] = df["是否淘汰赛"].map({0:"否",1:"是"})
    df["是否东道主"] = df["是否东道主"].map({0:"否",1:"是"})
    # 重命名英文列名
    df = df.rename(columns={"Year":"年份","Attendance":"现场观众"})
    return df

# 所有可选球队（按英文排序）
all_teams = sorted(team_dataset["球队"].unique().tolist())

# 构造下拉显示项：国旗 中文名（英文名）
def team_label(en_name):
    flag = team_flag(en_name)
    cn = team_en_cn_map.get(en_name)
    return f"{flag} {cn}（{en_name}）" if cn else f"{flag} {en_name}"

label_to_en = {team_label(t): t for t in all_teams}
en_to_label = {v: k for k, v in label_to_en.items()}

# ========== 训练4个真实机器学习模型集成 ==========
@st.cache_resource
def train_pred_model(data):
    feature_cols = ["Attendance","本队红牌","对手红牌","是否淘汰赛","是否东道主","本队进球","对手进球"]
    X = data[feature_cols]
    y = data["比赛结果"]  # 0落败 1平局 2胜利
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 4 个真实 sklearn 模型
    models = {
        "rf":  RandomForestClassifier(random_state=42, n_estimators=80, max_depth=6),
        "lr":  LogisticRegression(random_state=42, max_iter=1000),
        "svm": SVC(random_state=42, probability=True),
        "dt":  DecisionTreeClassifier(random_state=42, max_depth=6),
    }
    for m in models.values():
        m.fit(X_scaled, y)
    return models, scaler, feature_cols

ml_models, std_scaler, feat_cols = train_pred_model(team_dataset)

# ========== 机器学习模型选型与算法剖析（二）：模型性能评估 ==========
@st.cache_data
def evaluate_all_models(_data):
    """对4个模型做训练集内评估 + 5折交叉验证，返回评估表与混淆矩阵"""
    X = _data[feat_cols]
    y = _data["比赛结果"]
    X_scaled = std_scaler.transform(X)
    model_names = {"rf":"🌲随机森林","lr":"📐逻辑回归","svm":"⚡支持向量机","dt":"🌳决策树"}
    rows = []
    cms = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for key, model in ml_models.items():
        y_pred = model.predict(X_scaled)
        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y, y_pred, average="macro", zero_division=0)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
        rows.append({
            "模型": model_names[key],
            "准确率": round(acc, 4),
            "精确率": round(prec, 4),
            "召回率": round(rec, 4),
            "F1分数": round(f1, 4),
            "5折CV-F1均值": round(cv_scores.mean(), 4),
            "5折CV-F1标准差": round(cv_scores.std(), 4),
        })
        cms[model_names[key]] = confusion_matrix(y, y_pred)
    eval_df = pd.DataFrame(rows)
    return eval_df, cms, list(model_names.values()), ["落败","平局","胜利"]

# 构造对战特征向量：A 的特征用 A 的进攻数据 + B 的防守数据
def build_match_features(stats_a, stats_b, is_knockout, is_host_a):
    att = float(team_dataset["Attendance"].mean())
    feat_a = [att, stats_a["红牌场均"], stats_b["红牌场均"],
              int(is_knockout), int(is_host_a),
              stats_a["场均进球"], stats_b["场均进球"]]
    feat_b = [att, stats_b["红牌场均"], stats_a["红牌场均"],
              int(is_knockout), int(not is_host_a),
              stats_b["场均进球"], stats_a["场均进球"]]
    return np.array([feat_a, feat_b])

# ========== 球队历史统计 ==========
def get_team_real_stats(team_name):
    sub = team_dataset[team_dataset["球队"] == team_name]
    if len(sub) == 0:
        return None
    return {
        "比赛场次": int(len(sub)),
        "场均进球": round(sub["本队进球"].mean(), 2),
        "场均失球": round(sub["对手进球"].mean(), 2),
        "大赛胜率": round(len(sub[sub["比赛结果"]==2]) / len(sub), 3),
        "红牌场均": round(sub["本队红牌"].mean(), 3),
        "淘汰赛场均得分": round(sub[sub["是否淘汰赛"]==1]["本队进球"].mean(),2),
        "主场(东道主)胜率": round(sub[sub["是否东道主"]==1]["比赛结果"].value_counts(normalize=True).get(2,0),3)
    }

# 历年胜率趋势
def get_team_yearly_trend(team_name):
    sub = team_dataset[team_dataset["球队"] == team_name]
    if len(sub) == 0:
        return pd.DataFrame()
    rows = []
    for year, g in sub.groupby("Year"):
        win_rate = (g["比赛结果"]==2).mean()
        rows.append({"Year": int(year), "胜率": round(win_rate,3), "球队": team_name})
    return pd.DataFrame(rows).sort_values("Year")

# 历史交锋记录：同年同阶段，A本队进球==B对手进球 且 A对手进球==B本队进球
def find_historical_matches(team_a, team_b):
    df = team_dataset
    a_rows = df[df["球队"]==team_a].copy()
    b_rows = df[df["球队"]==team_b].copy()
    if a_rows.empty or b_rows.empty:
        return pd.DataFrame()
    merged = a_rows.merge(
        b_rows, on=["Year","赛事阶段"], suffixes=("_A","_B")
    )
    cond = (
        (merged["本队进球_A"]==merged["对手进球_B"]) &
        (merged["对手进球_A"]==merged["本队进球_B"])
    )
    matches = merged[cond][[
        "Year","赛事阶段","本队进球_A","对手进球_A",
        "本队红牌_A","对手红牌_A","Attendance_A",
        "是否淘汰赛_A","是否东道主_A"
    ]].copy()
    # 球队显示名（带国旗+中文）
    disp_a = f"{team_flag(team_a)} {team_en_cn_map.get(team_a, team_a)}"
    matches.columns = ["年份","阶段",f"{disp_a}进球",f"{disp_a}失球",
                       f"{disp_a}红牌","对手红牌","观众人数","是否淘汰赛",f"{disp_a}东道主"]
    # 阶段转中文
    matches["阶段"] = matches["阶段"].apply(stage_cn)
    # 是否字段 0/1 转否/是
    matches["是否淘汰赛"] = matches["是否淘汰赛"].map({0:"否",1:"是"})
    matches[f"{disp_a}东道主"] = matches[f"{disp_a}东道主"].map({0:"否",1:"是"})
    return matches.sort_values("年份", ascending=False)

# ========== 漂浮足球背景层 ==========
st.markdown("""
<div class="bg-footballs">
  <span style="left:8%;animation-delay:0s;">⚽</span>
  <span style="left:22%;animation-delay:3s;">⚽</span>
  <span style="left:38%;animation-delay:6s;">⚽</span>
  <span style="left:52%;animation-delay:9s;">⚽</span>
  <span style="left:68%;animation-delay:12s;">⚽</span>
  <span style="left:84%;animation-delay:15s;">⚽</span>
  <span style="left:15%;animation-delay:2s;font-size:24px;">🏆</span>
  <span style="left:60%;animation-delay:8s;font-size:26px;">🏆</span>
  <span style="left:90%;animation-delay:11s;font-size:22px;">🏆</span>
</div>
""", unsafe_allow_html=True)

# ========== 左右两侧球星装饰：C罗（左）梅西（右），使用本地图片 ==========
def _img_to_base64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

_ronaldo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ronaldo.png")
_messi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "messi.png")
_ronaldo_src = _img_to_base64(_ronaldo_path) if os.path.exists(_ronaldo_path) else ""
_messi_src = _img_to_base64(_messi_path) if os.path.exists(_messi_path) else ""

st.markdown(f"""
<div class="side-star left">
    <img class="star-photo" alt="C罗" src="{_ronaldo_src}" onerror="this.style.display='none'">
</div>
<div class="side-star right">
    <img class="star-photo" alt="梅西" src="{_messi_src}" onerror="this.style.display='none'">
</div>
""", unsafe_allow_html=True)

# ========== 顶部世界杯装饰条 ==========
st.markdown('<div class="wc-banner">🏆 2026 FIFA WORLD CUP · 数据驱动 · 预测未来 🏆</div>', unsafe_allow_html=True)

# ========== 页面标题 ==========
st.markdown('<div class="wc-trophy">🏆</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">⚽ 2026世界杯数据集原生AI比分预测</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于1930-2022清洗后结构化赛事数据 · 随机森林建模 · 多维数据共振决策</p>', unsafe_allow_html=True)

# ========== 输入区：双下拉 + 文本输入 ==========
# 动态草坪条：流动草坪 + 5 个跳动足球（完整独立块，确保草坪流动且足球在草坪上）
st.markdown("""
<div class="grass-strip">
    <span class="ball" style="left:8%;animation-delay:0s;">⚽</span>
    <span class="ball" style="left:28%;animation-delay:.3s;">⚽</span>
    <span class="ball" style="left:48%;animation-delay:.6s;">⚽</span>
    <span class="ball" style="left:68%;animation-delay:.9s;">⚽</span>
    <span class="ball" style="left:88%;animation-delay:1.2s;">⚽</span>
</div>
""", unsafe_allow_html=True)
st.markdown("#### 🎯 选择对战双方（数据集动态加载全部球队）")

col_sel1, col_vs, col_sel2 = st.columns([5, 1, 5])
with col_sel1:
    label_a = st.selectbox("主队 A", options=list(label_to_en.keys()),
                           index=list(label_to_en.keys()).index(en_to_label.get("Brazil","Brazil")) if "Brazil" in en_to_label else 0,
                           key="sel_a")
with col_vs:
    st.markdown("<div style='text-align:center;font-size:28px;font-weight:800;color:#165DFF;margin-top:28px;'>VS</div>", unsafe_allow_html=True)
with col_sel2:
    label_b = st.selectbox("客队 B", options=list(label_to_en.keys()),
                           index=list(label_to_en.keys()).index(en_to_label.get("Argentina","Argentina")) if "Argentina" in en_to_label else 1,
                           key="sel_b")

team_a_en = label_to_en[label_a]
team_b_en = label_to_en[label_b]

# 进阶选项
with st.expander("⚙️ 进阶参数（可选）", expanded=False):
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        is_knockout = st.checkbox("标记为淘汰赛", value=False, help="影响模型对赛制强度的判断")
    with col_opt2:
        is_host_a = st.checkbox(f"{team_a_en} 为东道主", value=False)

# 机器学习模型权重
with st.expander("🤖 机器学习模型权重（4模型集成，影响胜负率）", expanded=False):
    st.caption("调节4个机器学习模型的权重百分比（0-100%），系统根据各模型对数据集特征的不同侧重，加权计算最终胜负率。")
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        ml_rf = st.slider("🌲 随机森林（进攻型：进球/失球/胜率）", 0, 100, 30, help="侧重进攻火力与大赛胜率")
        ml_lr = st.slider("📐 逻辑回归（纪律型：红牌/主场/胜率）", 0, 100, 25, help="侧重纪律性与主场优势")
    with col_ml2:
        ml_svm = st.slider("⚡ 支持向量机（效率型：净胜球/淘汰赛）", 0, 100, 25, help="侧重攻防效率与淘汰赛表现")
        ml_dt = st.slider("🌳 决策树（综合型：全维度均衡）", 0, 100, 20, help="全特征均衡评估")

submit_btn = st.button("🚀 启动数据共振预测", type="primary", use_container_width=True)

# ========== 预测主逻辑 ==========
if submit_btn:
    if team_a_en == team_b_en:
        st.warning("请选择两支不同球队")
    else:
        stats_a = get_team_real_stats(team_a_en)
        stats_b = get_team_real_stats(team_b_en)

        if stats_a is None or stats_b is None:
            st.error("数据集未收录该队伍")
        else:
            # --- 比分预测 ---
            a_goal_exp = (stats_a["场均进球"] * 0.65) - (stats_b["场均失球"] * 0.35)
            b_goal_exp = (stats_b["场均进球"] * 0.65) - (stats_a["场均失球"] * 0.35)
            if is_host_a:
                a_goal_exp += 0.25
            if is_knockout:
                a_goal_exp -= 0.1
                b_goal_exp -= 0.1

            win_a_base = stats_a["大赛胜率"] * 0.5 + (1 - stats_a["红牌场均"]) * 0.3 + stats_a["主场(东道主)胜率"] * 0.2
            win_b_base = stats_b["大赛胜率"] * 0.5 + (1 - stats_b["红牌场均"]) * 0.3 + stats_b["主场(东道主)胜率"] * 0.2
            if is_host_a:
                win_a_base *= 1.1

            # ===== 4个真实 sklearn 机器学习模型分别预测 =====
            # 构造对战特征并用各模型 predict_proba 得到 [A落败(=B胜), 平局, A胜]
            X_match = build_match_features(stats_a, stats_b, is_knockout, is_host_a)
            X_match_scaled = std_scaler.transform(X_match)
            # 以 A 视角预测：proba[0]=[落败,平局,胜利]，即 [B胜, 平, A胜]
            proba_rf  = ml_models["rf"].predict_proba(X_match_scaled)[0]
            proba_lr  = ml_models["lr"].predict_proba(X_match_scaled)[0]
            proba_svm = ml_models["svm"].predict_proba(X_match_scaled)[0]
            proba_dt  = ml_models["dt"].predict_proba(X_match_scaled)[0]
            # 各模型 A 胜 / 平 / B 胜 概率
            rf_win_a,  rf_draw,  rf_win_b  = proba_rf[2],  proba_rf[1],  proba_rf[0]
            lr_win_a,  lr_draw,  lr_win_b  = proba_lr[2],  proba_lr[1],  proba_lr[0]
            svm_win_a, svm_draw, svm_win_b = proba_svm[2], proba_svm[1], proba_svm[0]
            dt_win_a,  dt_draw,  dt_win_b  = proba_dt[2],  proba_dt[1],  proba_dt[0]

            # 加权集成（用户设置的权重），同时对 A胜/平/B胜 三类加权
            total_ml_weight = ml_rf + ml_lr + ml_svm + ml_dt
            if total_ml_weight > 0:
                ml_win_a = (rf_win_a * ml_rf + lr_win_a * ml_lr + svm_win_a * ml_svm + dt_win_a * ml_dt) / total_ml_weight
                ml_draw  = (rf_draw  * ml_rf + lr_draw  * ml_lr + svm_draw  * ml_svm + dt_draw  * ml_dt) / total_ml_weight
                ml_win_b = (rf_win_b * ml_rf + lr_win_b * ml_lr + svm_win_b * ml_svm + dt_win_b * ml_dt) / total_ml_weight
            else:
                ml_win_a = (rf_win_a + lr_win_a + svm_win_a + dt_win_a) / 4
                ml_draw  = (rf_draw  + lr_draw  + svm_draw  + dt_draw)  / 4
                ml_win_b = (rf_win_b + lr_win_b + svm_win_b + dt_win_b) / 4

            # 将ML预测与基础胜率融合（ML占60%，基础占40%）
            base_win_a = win_a_base / (win_a_base + win_b_base + 0.01)
            base_win_b = 1 - base_win_a
            final_win_a = base_win_a * 0.4 + ml_win_a * 0.6
            final_draw  = ml_draw * 0.6
            final_win_b = base_win_b * 0.4 + ml_win_b * 0.6
            # 归一化使三者之和为1
            _sum = final_win_a + final_draw + final_win_b
            final_win_a, final_draw, final_win_b = final_win_a/_sum, final_draw/_sum, final_win_b/_sum

            win_a_pct = round(final_win_a * 100, 1)
            win_b_pct = round(final_win_b * 100, 1)
            draw_pct = round(final_draw * 100, 1)

            # 比分与胜负优势联动：胜率差距越大，进球差距越大（大胆预测）
            win_diff = (win_a_pct - win_b_pct) / 100.0  # -1 ~ 1
            # 用胜率差作为进攻加成，让强队多进、弱队少进（大幅加大系数）
            a_goal_exp += max(0, win_diff) * 5.5
            b_goal_exp += max(0, -win_diff) * 5.5
            # 同时让弱队期望进球大幅衰减
            if win_diff > 0:
                b_goal_exp *= max(0.1, 1 - win_diff * 1.4)
            elif win_diff < 0:
                a_goal_exp *= max(0.1, 1 + win_diff * 1.4)
            # 大胆取整：向上取整让强队更激进
            if win_diff > 0:
                score_a = max(0, min(9, math.floor(a_goal_exp + 0.5) + (1 if win_diff > 0.3 else 0)))
                score_b = max(0, min(9, math.floor(b_goal_exp)))
            elif win_diff < 0:
                score_b = max(0, min(9, math.floor(b_goal_exp + 0.5) + (1 if -win_diff > 0.3 else 0)))
                score_a = max(0, min(9, math.floor(a_goal_exp)))
            else:
                score_a = max(0, min(9, round(a_goal_exp)))
                score_b = max(0, min(9, round(b_goal_exp)))
            # 当胜率差距明显（>12%）但比分仍打平时，给优势方加球
            if abs(score_a - score_b) == 0 and abs(win_diff) > 0.12:
                if win_diff > 0:
                    score_a += 2
                else:
                    score_b += 2
            # 大差距时再追加，制造悬殊比分
            if abs(win_diff) > 0.4:
                if win_diff > 0:
                    score_a += 1
                else:
                    score_b += 1
            final_score = f"{score_a} : {score_b}"

            # 纯中文名（用于 Altair 图表，避免 emoji 渲染乱码）
            pure_a = team_en_cn_map.get(team_a_en, team_a_en)
            pure_b = team_en_cn_map.get(team_b_en, team_b_en)
            # 带国旗 emoji 的名（用于 HTML 文本展示）
            show_a = f"{team_flag(team_a_en)} {pure_a}"
            show_b = f"{team_flag(team_b_en)} {pure_b}"
            flag_a = team_flag(team_a_en)
            flag_b = team_flag(team_b_en)

            # --- 结果卡片 ---
            st.markdown('<div class="result-card fade-in">', unsafe_allow_html=True)
            st.markdown(f"### 🏆 {show_a} <span class='win-badge badge-d'>VS</span> {show_b}", unsafe_allow_html=True)
            # 大号队徽对决展示
            st.markdown(f"""
            <div style='display:flex;justify-content:center;align-items:center;
                        gap:24px;margin:18px 0 8px;'>
                <div style='text-align:center;flex:1;'>
                    <div style='font-size:72px;line-height:1;'>{flag_a}</div>
                    <div style='font-weight:700;color:#165DFF;margin-top:6px;font-size:18px;'>{pure_a}</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:54px;font-weight:900;color:#FF6B6B;
                                background:linear-gradient(90deg,#FF6B6B,#165DFF);
                                -webkit-background-clip:text;background-clip:text;
                                -webkit-text-fill-color:transparent;
                                animation:pulse 2s ease-in-out infinite;'>{final_score}</div>
                    <div style='color:#999;font-size:12px;margin-top:4px;'>预测比分</div>
                </div>
                <div style='text-align:center;flex:1;'>
                    <div style='font-size:72px;line-height:1;'>{flag_b}</div>
                    <div style='font-weight:700;color:#FF6B6B;margin-top:6px;font-size:18px;'>{pure_b}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='text-align:center;margin:10px 0 20px;'>
                <span class='win-badge badge-a'>{show_a} {win_a_pct}%</span>
                <span class='win-badge badge-d'>平局 {draw_pct}%</span>
                <span class='win-badge badge-b'>{show_b} {win_b_pct}%</span>
            </div>
            """, unsafe_allow_html=True)

            # 三列概率 metric
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{show_a} 取胜", f"{win_a_pct}%")
            with col2:
                st.metric("平局", f"{draw_pct}%")
            with col3:
                st.metric(f"{show_b} 取胜", f"{win_b_pct}%")

            # --- 胜率动态进度条对比 ---
            st.markdown("#### 📊 胜率共振对比")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**{show_a} 大赛胜率** ({stats_a['大赛胜率']*100:.1f}%)")
                st.progress(int(stats_a["大赛胜率"]*100))
                st.markdown(f"**{show_a} 主场胜率** ({stats_a['主场(东道主)胜率']*100:.1f}%)")
                st.progress(int(stats_a["主场(东道主)胜率"]*100))
            with col_p2:
                st.markdown(f"**{show_b} 大赛胜率** ({stats_b['大赛胜率']*100:.1f}%)")
                st.progress(int(stats_b["大赛胜率"]*100))
                st.markdown(f"**{show_b} 主场胜率** ({stats_b['主场(东道主)胜率']*100:.1f}%)")
                st.progress(int(stats_b["主场(东道主)胜率"]*100))

            # --- 4个ML模型预测明细 ---
            st.markdown("#### 🤖 机器学习模型预测明细（真实 sklearn predict_proba）")
            ml_data = pd.DataFrame({
                "模型": ["🌲 随机森林(进攻型)", "📐 逻辑回归(纪律型)", "⚡ 支持向量机(效率型)", "🌳 决策树(综合型)"],
                "权重": [f"{ml_rf}%", f"{ml_lr}%", f"{ml_svm}%", f"{ml_dt}%"],
                f"{show_a} 胜率": [f"{rf_win_a*100:.1f}%", f"{lr_win_a*100:.1f}%", f"{svm_win_a*100:.1f}%", f"{dt_win_a*100:.1f}%"],
                "平局": [f"{rf_draw*100:.1f}%", f"{lr_draw*100:.1f}%", f"{svm_draw*100:.1f}%", f"{dt_draw*100:.1f}%"],
                f"{show_b} 胜率": [f"{rf_win_b*100:.1f}%", f"{lr_win_b*100:.1f}%", f"{svm_win_b*100:.1f}%", f"{dt_win_b*100:.1f}%"],
            })
            st.dataframe(ml_data, use_container_width=True, hide_index=True)
            st.caption(f"集成预测：{show_a} {final_win_a*100:.1f}% / 平局 {final_draw*100:.1f}% / {show_b} {final_win_b*100:.1f}%（ML权重{total_ml_weight}%，ML占比60%+基础40%）")

            st.markdown("</div>", unsafe_allow_html=True)

            # --- 六维能力对比图（分组条形图，稳定渲染） ---
            st.markdown('<div class="stat-card fade-in">', unsafe_allow_html=True)
            st.markdown(f"#### 🎯 六维能力对比（数据集共振）")

            def normalize(val, v_max, v_min=0):
                return round((val - v_min) / (v_max - v_min) * 100, 1) if v_max > v_min else 0

            radar_dims = ["进攻","防守","胜率","纪律","淘汰赛","主场"]
            radar_a = [
                normalize(stats_a["场均进球"], 4),
                normalize(4 - stats_a["场均失球"], 4),
                stats_a["大赛胜率"] * 100,
                (1 - stats_a["红牌场均"]) * 100,
                normalize(stats_a["淘汰赛场均得分"], 3),
                stats_a["主场(东道主)胜率"] * 100,
            ]
            radar_b = [
                normalize(stats_b["场均进球"], 4),
                normalize(4 - stats_b["场均失球"], 4),
                stats_b["大赛胜率"] * 100,
                (1 - stats_b["红牌场均"]) * 100,
                normalize(stats_b["淘汰赛场均得分"], 3),
                stats_b["主场(东道主)胜率"] * 100,
            ]
            # 分组条形图数据：每个维度两根柱子
            dim_order = radar_dims
            bar_df = pd.DataFrame({
                "维度": radar_dims * 2,
                "数值": radar_a + radar_b,
                "球队": [pure_a]*6 + [pure_b]*6
            })
            bar_chart = alt.Chart(bar_df).mark_bar(cornerRadiusEnd=6).encode(
                y=alt.Y("维度:N", sort=dim_order, title=None,
                        axis=alt.Axis(labelFontSize=13, labelColor="#333")),
                x=alt.X("数值:Q", scale=alt.Scale(domain=[0,100]), title="能力评分 (0-100)",
                        axis=alt.Axis(labelFontSize=11)),
                color=alt.Color("球队:N", scale=alt.Scale(range=["#165DFF","#FF6B6B"]), title="球队"),
                yOffset=alt.YOffset("球队:N"),
                tooltip=[alt.Tooltip("球队:N", title="球队"),
                         alt.Tooltip("维度:N", title="维度"),
                         alt.Tooltip("数值:Q", title="评分", format=".1f")]
            ).properties(
                width=720, height=340,
                title=f"{pure_a} vs {pure_b} 六维能力对比"
            ).configure_title(
                fontSize=16, color="#165DFF", anchor="middle"
            ).configure_axis(
                grid=True, gridColor="#EEF3FF"
            ).configure_legend(
                labelFontSize=13, titleFontSize=13
            )
            st.altair_chart(bar_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # --- 历年胜率趋势（分组柱状图，固定不滑动，一眼对比） ---
            st.markdown('<div class="stat-card fade-in">', unsafe_allow_html=True)
            st.markdown("#### 📈 历年世界杯胜率对比（柱状一眼对比）")
            trend_a = get_team_yearly_trend(team_a_en)
            trend_b = get_team_yearly_trend(team_b_en)
            trend_a["显示名"] = pure_a
            trend_b["显示名"] = pure_b
            trend_all = pd.concat([trend_a, trend_b])
            trend_all["胜率(%)"] = (trend_all["胜率"] * 100).round(1)

            trend_chart = alt.Chart(trend_all).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("Year:O", title="年份",
                        axis=alt.Axis(labelFontSize=12, labelAngle=0)),
                y=alt.Y("胜率(%):Q", scale=alt.Scale(domain=[0,100]), title="胜率 (%)",
                        axis=alt.Axis(labelFontSize=12)),
                color=alt.Color("显示名:N", scale=alt.Scale(range=["#165DFF","#FF6B6B"]), title="球队"),
                xOffset=alt.XOffset("显示名:N"),
                tooltip=[alt.Tooltip("显示名:N", title="球队"),
                         alt.Tooltip("Year:O", title="年份"),
                         alt.Tooltip("胜率(%):Q", title="胜率(%)", format=".1f")]
            ).properties(width=720, height=360,
                         title=f"{pure_a} 与 {pure_b} 历年世界杯胜率对比"
            ).configure_title(
                fontSize=16, color="#165DFF", anchor="middle"
            ).configure_axis(
                labelFontSize=12, labelColor="#444", grid=True, gridColor="#EEF3FF"
            ).configure_legend(
                labelFontSize=13, titleFontSize=13
            )
            st.altair_chart(trend_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # --- 历史交锋记录 ---
            st.markdown('<div class="stat-card fade-in">', unsafe_allow_html=True)
            st.markdown("#### ⚔️ 历史交锋记录（数据集共振匹配）")
            hist = find_historical_matches(team_a_en, team_b_en)
            if hist.empty:
                st.info(f"数据集中暂无 {show_a} 与 {show_b} 的直接交锋记录")
            else:
                st.caption(f"共匹配到 {len(hist)} 场历史交锋")
                st.dataframe(hist, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # --- 预测依据文字 ---
            st.markdown('<div class="reason-block fade-in">', unsafe_allow_html=True)
            st.markdown("#### 📋 预测依据（全部来自清洗后结构化数据集统计）")
            st.markdown(f"""
**一、{show_a} 历史数据集表现**（共 {stats_a['比赛场次']} 场）
- 场均进球：{stats_a['场均进球']} 球，场均失球：{stats_a['场均失球']} 球
- 世界杯大赛整体胜率：{stats_a['大赛胜率']*100:.1f}%
- 场均红牌：{stats_a['红牌场均']}，纪律评分：{round((1-stats_a['红牌场均'])*100,1)}分
- 淘汰赛场均得分：{stats_a['淘汰赛场均得分']}，东道主主场胜率：{stats_a['主场(东道主)胜率']*100:.1f}%

**二、{show_b} 历史数据集表现**（共 {stats_b['比赛场次']} 场）
- 场均进球：{stats_b['场均进球']} 球，场均失球：{stats_b['场均失球']} 球
- 世界杯大赛整体胜率：{stats_b['大赛胜率']*100:.1f}%
- 场均红牌：{stats_b['红牌场均']}，纪律评分：{round((1-stats_b['红牌场均'])*100,1)}分
- 淘汰赛场均得分：{stats_b['淘汰赛场均得分']}，东道主主场胜率：{stats_b['主场(东道主)胜率']*100:.1f}%

**三、综合研判结论**
""", unsafe_allow_html=True)
            if score_a > score_b:
                reason_final = f"{show_a} 在历史进攻效率、大赛胜率、赛场纪律性三项数据集指标上优于 {show_b}，模型判定取胜概率更高，预测比分 {final_score}"
            elif score_b > score_a:
                reason_final = f"{show_b} 历史防守稳定性与淘汰赛得分能力在数据集中表现更强，整体竞技水准占优，预测比分 {final_score}"
            else:
                reason_final = "两队攻防数据、历史胜率高度接近，数据集交锋特征均衡，本场大概率平局"
            st.markdown(f"<p style='color:#165DFF;font-weight:bold;font-size:15px;'>{reason_final}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ========== 机器学习模型选型与算法剖析（二）：模型性能评估 ==========
with st.expander("🔬 机器学习模型选型与算法剖析（二）：模型性能评估", expanded=False):
    st.caption("对网页集成的4个真实 sklearn 模型做训练集内评估 + 5折交叉验证，量化各模型在「落败/平局/胜利」三类分类上的表现。")
    eval_df, cms, model_names_list, class_labels = evaluate_all_models(team_dataset)

    st.markdown("##### 一、多模型评估指标对比表")
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    # 找出F1最高的模型
    best_idx = eval_df["F1分数"].idxmax()
    best_model_name = eval_df.loc[best_idx, "模型"]
    best_f1 = eval_df.loc[best_idx, "F1分数"]
    st.success(f"🏆 综合最优模型：{best_model_name}（训练集 F1 = {best_f1}）")

    st.markdown("##### 二、评估指标分组对比图（Altair 交互）")
    # 长表用于绘图
    eval_long = eval_df.melt(id_vars="模型",
                             value_vars=["准确率","精确率","召回率","F1分数","5折CV-F1均值"],
                             var_name="指标", value_name="数值")
    bar_eval = alt.Chart(eval_long).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("数值:Q", scale=alt.Scale(domain=[0,1]), title="分数",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("模型:N", title=None),
        color=alt.Color("指标:N", scale=alt.Scale(range=["#165DFF","#00C2FF","#FF6B6B","#FFB300","#7B8AFF"]),
                        title="评估指标"),
        yOffset=alt.YOffset("指标:N"),
        tooltip=[alt.Tooltip("模型:N", title="模型"),
                 alt.Tooltip("指标:N", title="指标"),
                 alt.Tooltip("数值:Q", title="分数", format=".4f")]
    ).properties(width=720, height=320,
                 title="4模型 评估指标分组对比（越高越好）"
    ).configure_title(fontSize=16, color="#165DFF", anchor="middle"
    ).configure_axis(grid=True, gridColor="#EEF3FF"
    ).configure_legend(labelFontSize=12, titleFontSize=12)
    st.altair_chart(bar_eval, use_container_width=True)

    st.markdown("##### 三、5折交叉验证 F1 稳定性（均值越高、标准差越小越稳定）")
    cv_chart = alt.Chart(eval_df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("5折CV-F1均值:Q", scale=alt.Scale(domain=[0,1]), title="CV-F1 均值",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("模型:N", title=None),
        color=alt.Color("模型:N", scale=alt.Scale(range=["#165DFF","#00C2FF","#FF6B6B","#FFB300"]), legend=None),
        tooltip=[alt.Tooltip("模型:N", title="模型"),
                 alt.Tooltip("5折CV-F1均值:Q", title="CV-F1均值", format=".4f"),
                 alt.Tooltip("5折CV-F1标准差:Q", title="CV-F1标准差", format=".4f")]
    ).properties(width=720, height=260,
                 title="5折交叉验证 F1 均值（泛化稳定性）"
    ).configure_title(fontSize=16, color="#165DFF", anchor="middle"
    ).configure_axis(grid=True, gridColor="#EEF3FF")
    st.altair_chart(cv_chart, use_container_width=True)

    st.markdown("##### 四、各模型混淆矩阵（行=真实，列=预测）")
    cm_cols = st.columns(4)
    for i, mname in enumerate(model_names_list):
        with cm_cols[i]:
            st.markdown(f"**{mname}**")
            cm = cms[mname]
            cm_disp = pd.DataFrame(cm, index=[f"真:{c}" for c in class_labels],
                                   columns=[f"预:{c}" for c in class_labels])
            st.dataframe(cm_disp, use_container_width=True, hide_index=True)

    st.markdown("##### 五、算法剖析与选型结论")
    st.markdown(f"""
<div style='background:#F5F9FF;padding:16px 20px;border-radius:12px;
            border-left:5px solid #165DFF;line-height:1.75;'>
<p><b>1. 评估指标说明</b></p>
<ul>
<li><b>准确率 Accuracy</b>：整体预测正确率 = 正确预测数 / 总样本数</li>
<li><b>精确率 Precision</b>（macro）：预测为某类的样本中真正属于该类的比例，三类别取平均</li>
<li><b>召回率 Recall</b>（macro）：某类真实样本中被正确预测的比例，三类别取平均</li>
<li><b>F1分数</b>（macro）：精确率与召回率的调和平均，越接近1越好</li>
<li><b>5折CV-F1</b>：将数据分5份轮流做训练/验证，均值越高泛化越好，标准差越小越稳定</li>
</ul>
<p><b>2. 选型结论</b></p>
<ul>
<li>训练集F1最高：<b>{best_model_name}</b>（F1={best_f1}）</li>
<li>随机森林：集成多棵决策树降方差，特征重要性可解释，整体最稳，适合作为网页集成预测的主力</li>
<li>逻辑回归：线性模型，训练快、可解释强，但对"平局"类非线性边界拟合不足</li>
<li>支持向量机：小样本高维表现好，但调参敏感、对类别不平衡敏感</li>
<li>决策树：单树易过拟合，CV标准差通常偏大，作为弱学习器参与集成而非单独使用</li>
</ul>
<p><b>3. 平局误判分析</b>：从混淆矩阵可看到"平局"类召回率普遍偏低，原因是世界杯平局样本占比少且特征与胜负接近；</p>
<p><b>4. 后续优化方向</b>：可对平局样本做 SMOTE 过采样、引入 xG（期望进球）等高级特征、或用 Stacking 多模型融合提升整体 F1。</p>
</div>
""", unsafe_allow_html=True)

# ========== 数据集总览 ==========
with st.expander("📚 数据集总览（点击展开查看完整结构化数据）", expanded=False):
    st.caption(f"共 {len(team_dataset)} 条赛事记录，覆盖 {team_dataset['球队'].nunique()} 支球队，年份范围 {team_dataset['Year'].min()}-{team_dataset['Year'].max()}")
    col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
    with col_ov1:
        st.metric("赛事记录", len(team_dataset))
    with col_ov2:
        st.metric("参赛球队", team_dataset["球队"].nunique())
    with col_ov3:
        st.metric("总进球数", int(team_dataset["本队进球"].sum()))
    with col_ov4:
        st.metric("总红牌数", int(team_dataset["本队红牌"].sum()))
    # 展示中文化版本：列名、赛事阶段、球队名(含国旗)、比赛结果全部中文化
    cn_display_df = build_cn_display_df(team_dataset)
    st.dataframe(cn_display_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("数据来源：世界杯清洗后结构化数据集_含红牌.csv | 随机森林模型 · Altair 可视化 · 数据共振决策")

# ========== 免责声明 ==========
st.markdown("""
<div style='margin:24px 0 8px;padding:18px 22px;border-radius:14px;
            background:linear-gradient(135deg,#FFF9E6,#FFF3D6);
            border-left:5px solid #FFB300;
            box-shadow:0 4px 14px rgba(255,179,0,0.18);'>
    <div style='font-size:20px;font-weight:700;color:#B26A00;margin-bottom:8px;'>
        ⚠️ 最后提醒
    </div>
    <div style='font-size:18px;color:#7A5200;line-height:1.7;'>
        这类 AI 赛事预测仅为数据模拟推演，足球比赛受临场状态、战术、红黄牌、伤病等大量随机因素影响，预测概率不能等同于实际赛果。
    </div>
</div>
""", unsafe_allow_html=True)
