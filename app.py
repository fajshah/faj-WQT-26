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
        bill_ids = df_l1["bill_id"].tolist()
        selected_bill = st.selectbox("Select Bill ID", bill_ids)
        
        row = df_l1[df_l1["bill_id"] == selected_bill].iloc[0]
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Bill Image Preview")
            from pathlib import Path
            selected_bill_id = selected_bill
            
            # Search directories where bills might exist
            possible_dirs = [
                Path("test/bills"),
                Path("train_bills"),
                Path("bills"),
                Path("test"),
                Path("data/bills"),
                Path("output")
            ]
            
            image_path = None
            extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
            
            for directory in possible_dirs:
                if directory.exists():
                    for ext in extensions:
                        candidate = directory / f"{selected_bill_id}{ext}"
                        if candidate.is_file():
                            image_path = str(candidate)
                            break
                if image_path:
                    break
            
            if not image_path:
                # Fallback to any available image in workspace if specific file is absent
                all_imgs = glob.glob("train_bills/*.*") + glob.glob("test/bills/*.*") + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.png", recursive=True)
                if all_imgs:
                    image_path = all_imgs[0]
            
            if image_path:
                st.image(image_path, use_container_width=True)
            else:
                st.warning("Image not found for this bill ID.")
                
        with col2:
            st.subheader("Extracted JSON")
            json_str = row["json"]
            if pd.isna(json_str) or json_str.strip() == "":
                st.info("No JSON extracted yet or extraction pending.")
            else:
                st.json(json_str)
        
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
