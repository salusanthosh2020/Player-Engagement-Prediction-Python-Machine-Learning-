import streamlit as st
import pickle
from PIL import Image
import numpy as np

def main():
    st.set_page_config(
        page_title="Gaming Engagement Predictor",
        page_icon="🎮",
        layout="wide"
    )
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), 
                    url("https://images.unsplash.com/photo-1542751371-adc38448a05e?ixlib=rb-1.2.1&auto=format&fit=crop&w=2850&q);
        background-size: cover;
        color: #00ff41; 
    }=80"
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(20, 20, 25, 0.85);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e3e42;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3.5em;
        background-color: #6200ea;
        color: white;
        font-weight: bold;
        border: 2px solid #bb86fc;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .stButton>button:hover {
        background-color: #bb86fc;
        color: black;
        box-shadow: 0 0 20px #bb86fc;
    }

    label {
        color: #bb86fc !important;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
    }
    .stSuccess {
        background-color: rgba(25, 135, 84, 0.2);
        border: 1px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; color: #00fbff; text-shadow: 2px 2px #ff0055;'>🕹️ PLAYER ENGAGEMENT PROTOCOL</h1>",
        unsafe_allow_html=True)
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.markdown("![Gaming](https://img.icons8.com/neon/256/controller.png)")
    with col_text:
        st.info("🧬 **SYSTEM STATUS:** Ready for Data Input. Analysis parameters initialized.")
        st.write("---")
    with st.container():
        st.markdown("### 🛠️ METRIC CALIBRATION")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            Genre = st.selectbox("👾 GAME GENRE", ['Strategy', 'Sports', 'Action', 'RPG', 'Simulation'])
            InGamePurchases = st.radio("💎 MICROTRANSACTIONS", ['Yes', 'No'], horizontal=True)
        with row1_col2:
            GameDifficulty = st.select_slider("🔥 DIFFICULTY LEVEL", options=['Easy', 'Medium', 'Hard'])
            SessionsPerWeek = st.number_input("📅 SESSIONS / WEEK", min_value=0, max_value=168, value=5)
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            AvgSessionDurationMinutes = st.number_input("⏱️ AVG DURATION (MINS)", min_value=1, value=30)
            PlayerLevel = st.number_input("🆙 PLAYER LEVEL", min_value=1, value=1)
        with row2_col2:
            AchievementsUnlocked = st.number_input("🏆 ACHIEVEMENTS", min_value=0, value=0)
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        encoder = pickle.load(open('ENCODER(correct).sav', 'rb'))
        scaler = pickle.load(open('scaler(correct).sav', 'rb'))
        model = pickle.load(open('model_rf(correct).sav', 'rb'))
        if st.button('🚀 RUN NEURAL ANALYSIS'):
            GameGenre_enc = encoder[2].transform([Genre]).item()
            igp_enc = encoder[3].transform([InGamePurchases]).item()
            Dif_enc = encoder[4].transform([GameDifficulty]).item()
            features = [GameGenre_enc, igp_enc, Dif_enc, SessionsPerWeek,
                        AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked]
            result = model.predict(scaler.transform([features]))
            st.write("---")
            st.subheader("📊 SCAN RESULT:")
            if result == 2:
                st.balloons()
                st.markdown("""
                    <div style="background-color:rgba(0, 255, 65, 0.1); padding: 20px; border: 2px solid #00ff41; border-radius: 10px;">
                        <h2 style="color: #00ff41; margin:0;">RANK: S-TIER (HIGH ENGAGEMENT)</h2>
                        <p>Core player detected. Loyalty metrics are off the charts!</p>
                    </div>
                """, unsafe_allow_html=True)
            elif result == 0:
                st.markdown("""
                    <div style="background-color:rgba(255, 0, 0, 0.1); padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px;">
                        <h2 style="color: #ff4b4b; margin:0;">RANK: CHURN RISK (LOW ENGAGEMENT)</h2>
                        <p>Warning: Player activity is dropping. Deploy re-engagement incentives.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color:rgba(255, 165, 0, 0.1); padding: 20px; border: 2px solid #ffa500; border-radius: 10px;">
                        <h2 style="color: #ffa500; margin:0;">RANK: B-TIER (MEDIUM ENGAGEMENT)</h2>
                        <p>Casual player behavior. Stable, but lacks "Whale" activity levels.</p>
                    </div>
                """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("📡 ERROR: Data files missing from the server.")
    except Exception as e:
        st.error(f"🛸 GLITCH DETECTED: {e}")


if __name__ == "__main__":
    main()
