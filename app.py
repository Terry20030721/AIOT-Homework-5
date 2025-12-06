import streamlit as st
import pandas as pd
from transformers import pipeline
import io

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI Writing Detector", page_icon="🤖", layout="wide")

st.title("🤖 AI 寫作偵測器 (AI Detector)")
st.write("一個基於 Transformer 模型的分類工具，用於判斷文本由 AI 生成還是人類撰寫。")

# --- 2. 模型選擇 ---
# 新增：定義可用的模型
MODELS = {
    "RoBERTa (for GPT-2)": "roberta-base-openai-detector",
    "RoBERTa (for ChatGPT)": "Hello-SimpleAI/chatgpt-detector-roberta"
}

selected_model_name = st.selectbox(
    "請選擇偵測模型:",
    options=list(MODELS.keys()),
    help="不同的模型對不同類型的 AI 生成文本有不一樣的偵測效果。"
)
selected_model_path = MODELS[selected_model_name]

# --- 3. 說明與模型限制 ---
with st.expander("ℹ️ 關於此工具與模型限制"):
    st.markdown(f"""
        - **目前模型**: `{selected_model_path}`
        - **模型背景**: `roberta-base-openai-detector` 是針對 GPT-2 時代的文本訓練的，而 `chatgpt-detector-roberta` 則對較新的模型（如 ChatGPT）有較好的偵測能力。
        - **使用建議**: 請將此工具的結果視為參考，而非最終定論。建議輸入 50 字以上的英文文本以獲得較佳效果。
    """)

# --- 4. 載入與快取模型/分析函式 ---
# 新增：@st.cache_resource 會快取模型本身，避免重複下載
@st.cache_resource
def load_model(model_path):
    return pipeline("text-classification", model=model_path)

# 新增：@st.cache_data 會快取函式的輸出結果
@st.cache_data
def run_prediction(_model, text_to_analyze):
    # _model 參數是為了確保當模型更換時，快取會失效
    # 實際執行時，我們從外部傳入已經載入的模型
    if not text_to_analyze or len(text_to_analyze.split()) < 10:
        return None
    try:
        # 加入 truncation=True，遇到太長的文章自動切斷，不會報錯
        return classifier(text_to_analyze, truncation=True, max_length=512)[0]
    except Exception:
        return {"label": "Error", "score": 0.0}

with st.spinner(f'正在載入模型: {selected_model_path}...'):
    try:
        classifier = load_model(selected_model_path)
    except Exception as e:
        st.error(f"模型載入失敗，請檢查網路連線或模型名稱。錯誤: {e}")
        st.stop()

# --- 5. 功能分頁 ---
tab1, tab2 = st.tabs(["📝 單筆文字分析", "📂 批次檔案分析"])

# ========================== TAB 1: 單筆文字分析 ==========================
with tab1:
    # --- Session State 初始化 ---
    if 'single_result' not in st.session_state:
        st.session_state.single_result = None
    if 'single_user_input' not in st.session_state:
        st.session_state.single_user_input = ""

    # --- 範例文字 ---
    AI_EXAMPLE = "The sun is a star, a glowing ball of hot gas, primarily hydrogen and helium, at the center of our solar system. Its gravity holds the solar system together, keeping everything from the largest planets to the smallest particles of debris in its orbit."
    HUMAN_EXAMPLE = "Just got back from a weekend trip to the mountains. The air was so fresh and crisp, a welcome change from the city smog. We hiked for miles and spent the evening by a crackling campfire. My legs are sore, but my soul feels recharged."

    # --- 回呼函式 ---
    def analyze_single_text():
        text = st.session_state.single_user_input
        if not text or len(text.split()) < 10:
            st.toast("⚠️ 請輸入至少 10 個單字的英文文本。", icon="✍️")
            st.session_state.single_result = None
            return
        
        with st.spinner("分析中..."):
            prediction = run_prediction(selected_model_path, text) # 使用快取函式
            st.session_state.single_result = prediction

    def clear_single_text():
        st.session_state.single_user_input = ""
        st.session_state.single_result = None

    def load_single_example(text):
        st.session_state.single_user_input = text
        st.session_state.single_result = None

    # --- UI 佈局 ---
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.text_area(
            "請在下方輸入英文文本:", 
            key="single_user_input",
            height=250,
            placeholder="Enter text here to analyze..."
        )
        
        btn_cols = st.columns(4)
        btn_cols[0].button("🔍 開始偵測", type="primary", on_click=analyze_single_text, use_container_width=True)
        btn_cols[1].button("清除內容", on_click=clear_single_text, use_container_width=True)
        btn_cols[2].button("載入 AI 範例", on_click=load_single_example, args=(AI_EXAMPLE,), use_container_width=True)
        btn_cols[3].button("載入人類範例", on_click=load_single_example, args=(HUMAN_EXAMPLE,), use_container_width=True)

    with col2:
        st.subheader("📊 分析結果")
        result = st.session_state.single_result

        if result is None:
            st.info("分析結果將會顯示於此。")
        else:
            # 根據模型輸出調整標籤名稱
            # roberta-base-openai-detector -> Real, Fake
            # chatgpt-detector-roberta -> human, machine
            is_fake = (result['label'] == 'Fake' or result['label'] == 'machine')
            score = result['score']
            
            prob_ai = score if is_fake else (1 - score)
            prob_human = 1 - prob_ai

            if abs(prob_ai - 0.5) < 0.1:
                st.warning("模型信心度不高，結果僅供參考。")

            if prob_ai > 0.5:
                st.error(f"**判定：AI 生成**")
                st.metric(label="AI 可能性", value=f"{prob_ai:.1%}")
            else:
                st.success(f"**判定：人類撰寫**")
                st.metric(label="人類可能性", value=f"{prob_human:.1%}")
            
            chart_data = pd.DataFrame({"類別": ["AI", "Human"], "機率": [prob_ai, prob_human]})
            bar_color = "#FF4B4B" if prob_ai > 0.5 else "#28A745"
            st.bar_chart(chart_data.set_index("類別"), color=bar_color)

# ========================== TAB 2: 批次檔案分析 ==========================
with tab2:
    st.subheader("上傳 .txt 檔案進行多筆分析")
    uploaded_file = st.file_uploader(
        "每個換行會被視為一筆獨立的文本。", type=["txt"]
    )

    if uploaded_file is not None:
        # 讀取檔案
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        lines = [line.strip() for line in stringio.readlines() if line.strip()]
        
        st.info(f"檔案讀取成功！共偵測到 {len(lines)} 筆文本。")

        if st.button("🚀 開始批次分析", type="primary"):
            results_list = []
            progress_bar = st.progress(0, text="分析進度...")

            for i, line in enumerate(lines):
                prediction = run_prediction(selected_model_path, line) # 使用快取函式
                
                if prediction:
                    is_fake = (prediction['label'] == 'Fake' or prediction['label'] == 'machine')
                    prob_ai = prediction['score'] if is_fake else (1 - prediction['score'])
                    
                    results_list.append({
                        "文本": line,
                        "判定": "AI 生成" if prob_ai > 0.5 else "人類撰寫",
                        "AI 可能性": prob_ai,
                    })
                else:
                    results_list.append({
                        "文本": line,
                        "判定": "長度不足或錯誤",
                        "AI 可能性": 0.0,
                    })
                
                progress_bar.progress((i + 1) / len(lines), text=f"分析進度: {i+1}/{len(lines)}")

            st.success("批次分析完成！")
            
            # 顯示結果表格
            df = pd.DataFrame(results_list)
            st.dataframe(
                df,
                column_config={
                    "文本": st.column_config.TextColumn("文本", width="large"),
                    "判定": st.column_config.TextColumn("判定"),
                    "AI 可能性": st.column_config.ProgressColumn(
                        "AI 可能性",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                },
                use_container_width=True
            )