import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import torch, numpy as np, json, os
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit
from core.audit.attribute import attribute

st.set_page_config(layout="wide")
st.title("Ábyssos: Analyst Console")

@st.cache_resource
def load_system():
    # Load Model
    model = MFSR(4, 2, 4)
    model.load_state_dict(torch.load("abyssos_weights/best_finetuned.pt", map_location="cpu"))
    model.eval()
    
    # Load Thresholds
    with open("weights/taus.json") as f: taus = json.load(f)
    
    # Load 1 Tile
    data = np.load(list(Path("abyssos_data/train").glob("*.npz"))[0])
    tile = Tile(
        lr=torch.from_numpy(data['lrs']).unsqueeze(0),
        masks=torch.from_numpy(data['masks']).unsqueeze(0),
        shifts=torch.from_numpy(data['shifts']).unsqueeze(0),
        sar=torch.from_numpy(data['sar']).unsqueeze(0),
        clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0),
        meta={"dates": [f"Pass_0{i+1}" for i in range(data['lrs'].shape[0])]}
    )
    
    with torch.no_grad():
        ar = audit(model, tile, scale=4)
    return ar, taus

try:
    ar, taus = load_system()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Super-Resolved Image (Click to Audit)")
        value = streamlit_image_coordinates("ui_layers/1_finetuned_sr.png", key="sr")
    with col2:
        st.subheader("2. Evidence Map (Red=Hallucinated, Green=Verified)")
        st.image("ui_layers/2_finetuned_evidence.png")

    if value is not None:
        x, y = value["x"], value["y"]
        res = attribute(ar, x, y, taus, radius=8)
        
        st.success(f"**Target Coordinates:** (X: {x}, Y: {y})")
        st.info(f"**Evidence Class:** {res['evidence_class']}")
        st.write("**Supporting Passes:**", res['supporting_dates'])
        st.write("**Missing/Clouded Passes:**", res['missing_dates'])
        
except Exception as e:
    st.error(f"System not initialized. Run export_final_ui.py first. Error: {e}")