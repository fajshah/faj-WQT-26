import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image

def find_bill_image(bill_id: str):
    search_dirs = ["train_bills", "test/bills", "bills", "data/bills", "."]
    extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for d in search_dirs:
        for ext in extensions:
            path = os.path.join(d, f"{bill_id}{ext}")
            if os.path.exists(path):
                return path
    matches = glob.glob(f"**/{bill_id}.*", recursive=True)
    if matches:
        return matches[0]
    # Fallback to display any available image in train_bills if specific bill image is missing in sample repo
    fallback_imgs = glob.glob("train_bills/*.*") + glob.glob("test/bills/*.*") + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.png", recursive=True)
    if fallback_imgs:
        return fallback_imgs[0]
    return None

st.set_page_config(page_title="Utility Bill Decoder", page_icon="⚡", layout="wide")

st.title("⚡ WTQ 2026 - Utility Bill Decoder Viewer")
st.markdown("Explore extracted utility bill JSONs and Level 2 Q&A results.")

# Sidebar for navigation
tab = st.sidebar.radio("Navigation", ["Level 1: JSON Extraction", "Level 2: Q&A Results", "Bill Images Viewer"])

# Load CSVs
l1_path = "output/level1.csv"
l2_path = "output/level2.csv"

if tab == "Level 1: JSON Extraction":
    st.header("Level 1: Structured JSON Extraction")
    if os.path.exists(l1_path):
        df_l1 = pd.read_csv(l1_path)
        bill_ids_list = df_l1["bill_id"].tolist()
        
        st.sidebar.markdown("### 📋 Select Bill ID")
        selected_bill_id = st.sidebar.radio("Choose Bill", options=bill_ids_list, key="sidebar_bill_radio")
        
        filtered_row = df_l1[df_l1['bill_id'] == selected_bill_id]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"Bill Image Preview ({selected_bill_id})")
            from pathlib import Path
            
            extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]
            search_dirs = [Path("test/bills"), Path("train_bills"), Path("bills"), Path("data/bills"), Path(".")]
            
            matched_img = None
            for d in search_dirs:
                if d.exists():
                    for ext in extensions:
                        candidate = d / f"{selected_bill_id}{ext}"
                        if candidate.is_file():
                            matched_img = str(candidate)
                            break
                if matched_img:
                    break
                    
            if not matched_img:
                all_imgs = glob.glob("train_bills/*.*") + glob.glob("test/bills/*.*") + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.png", recursive=True)
                if all_imgs:
                    matched_img = all_imgs[0]

            if matched_img:
                st.image(matched_img, use_container_width=True)
            else:
                st.warning(f"Image not found for {selected_bill_id}")
                
        with col2:
            st.subheader(f"Extracted JSON ({selected_bill_id})")
            if not filtered_row.empty:
                raw_json = filtered_row.iloc[0]['json']
                if pd.isna(raw_json) or str(raw_json).strip() == "":
                    st.info("No JSON extracted yet or extraction pending.")
                else:
                    try:
                        parsed_json = json.loads(raw_json) if isinstance(raw_json, str) else json.loads(str(raw_json).replace("'", '"'))
                        st.json(parsed_json)
                    except Exception:
                        st.text(str(raw_json))
            else:
                st.info("No data found for selected bill.")
        
        st.markdown("---")
        st.subheader("All Level 1 Records")
        st.dataframe(df_l1, use_container_width=True)
    else:
        st.error("output/level1.csv not found! Run main.py first.")

elif tab == "Level 2: Q&A Results":
    st.header("Level 2: Bill Q&A Answers")
    if os.path.exists(l2_path):
        df_l2 = pd.read_csv(l2_path, on_bad_lines='skip')
        bill_ids = df_l2["bill_id"].unique().tolist()
        selected_bill = st.selectbox("Select Bill ID for Q&A", bill_ids)
        
        filtered_df = df_l2[df_l2["bill_id"] == selected_bill]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Bill Image")
            img_path = find_bill_image(selected_bill)
            if img_path:
                image = Image.open(img_path)
                st.image(image, caption=selected_bill, use_container_width=True)
            else:
                st.warning("Image not found for this bill ID.")
                
        with col2:
            st.subheader("Questions & Answers")
            for _, r in filtered_df.iterrows():
                st.markdown(f"**Q ({r['question_id']}):** {r['question']}")
                ans = r['answer']
                if pd.isna(ans) or str(ans).strip() == "":
                    st.markdown("*Answer: [Pending/Not Answered]*")
                else:
                    st.success(f"**Answer:** {ans}")
                st.markdown("---")
        
        st.subheader("Full Q&A Table")
        st.dataframe(df_l2, use_container_width=True)
    else:
        st.error("output/level2.csv not found! Run main.py first.")

else:
    st.header("Bill Images Gallery")
    all_imgs = glob.glob("train_bills/*.*") + glob.glob("test/bills/*.*")
    if all_imgs:
        cols = st.columns(3)
        for idx, img_path in enumerate(all_imgs):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info("No bill images found.")
