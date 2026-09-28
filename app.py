import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import torch, numpy as np, json, os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit
from core.audit.attribute import attribute

st.set_page_config(page_title="Ábyssos Console", layout="wide", initial_sidebar_state="collapsed")

# Professional Dark Mode Styling
st.markdown("""
    <style>
    .metric-box { padding: 15px; border-radius: 8px; background-color: #1E1E2E; margin-bottom: 10px; border-left: 4px solid #4C566A; }
    .verif { color: #A3BE8C; font-weight: bold; font-size: 18px; }
    .prior { color: #BF616A; font-weight: bold; font-size: 18px; }
    .coord { color: #88C0D0; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Ábyssos: Intelligence Analyst Console")
st.markdown("---")

@st.cache_resource
def load_system():
    model = MFSR(4, 2, 4)
    model.load_state_dict(torch.load("abyssos_weights/best_finetuned.pt", map_location="cpu"))
    model.eval()
    with open("weights/taus.json") as f: taus = json.load(f)
    
    data = np.load(list(Path("abyssos_data/train").glob("*.npz"))[0])
    tile = Tile(
        lr=torch.from_numpy(data['lrs']).unsqueeze(0), masks=torch.from_numpy(data['masks']).unsqueeze(0),
        shifts=torch.from_numpy(data['shifts']).unsqueeze(0), sar=torch.from_numpy(data['sar']).unsqueeze(0),
        clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0),
        meta={"dates": [f"Pass_0{i+1}" for i in range(data['lrs'].shape[0])]}
    )
    with torch.no_grad(): ar = audit(model, tile, scale=4)
    return ar, taus

try:
    ar, taus = load_system()
    
    # 3-Column Layout
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    
    with col1:
        st.subheader("📡 2.5m Super-Resolved Image")
        st.caption("Click any feature to interrogate the residual stack.")
        value = streamlit_image_coordinates("ui_layers/1_finetuned_sr.png", key="sr")
        
    with col2:
        st.subheader("🛡️ Truth Auditor Evidence Map")
        st.caption("🟢 Verified Physics | 🔴 Hallucinated (Prior-Only)")
        st.image("ui_layers/2_finetuned_evidence.png", use_container_width=True)

    with col3:
        st.subheader("🔎 Attribution Panel")
        if value is None:
            st.info("Awaiting analyst input... Click a pixel on the SR image.")
        else:
            x, y = value["x"], value["y"]
            res = attribute(ar, x, y, taus, radius=8)
            
            cls_color = "verif" if res['evidence_class'] == "VERIFIED" else "prior"
            
            st.markdown(f"""
            <div class="metric-box">
                <p><b>Target:</b> <span class="coord">X: {x} | Y: {y}</span></p>
                <p><b>Class:</b> <span class="{cls_color}">{res['evidence_class']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### ✅ Supporting Passes")
            if res['supporting_dates']:
                for d in res['supporting_dates']: st.markdown(f"- `{d}`")
            else:
                st.markdown("*None*")
                
            st.markdown("#### ☁️ Missing/Clouded")
            if res['missing_dates']:
                for d, reason in res['missing_dates'].items(): st.markdown(f"- `{d}` *( {reason} )*")
            else:
                st.markdown("*None*")

except Exception as e:
    st.error(f"System not initialized. Error: {e}")