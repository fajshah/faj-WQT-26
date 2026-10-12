import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image
from pathlib import Path

def find_bill_image(bill_id: str):
    search_dirs = ["test/bills", "train_bills", "bills", "data/bills", "."]
    extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for d in search_dirs:
        for ext in extensions:
            path = os.path.join(d, f"{bill_id}{ext}")
            if os.path.exists(path):
                return path
    matches = glob.glob(f"**/{bill_id}.*", recursive=True)
    if matches:
        return matches[0]
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
            
            current_id = str(selected_bill_id).strip()
            found_images = glob.glob(f"**/{current_id}.*", recursive=True)
            valid_images = [img for img in found_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if valid_images:
                st.image(valid_images[0], use_container_width=True)
            else:
                all_imgs = [img for img in glob.glob("**/*.*", recursive=True) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                matched_fallback = None
                prefix = current_id.split("_")[0].upper()
                for img in all_imgs:
                    if prefix in img.upper():
                        matched_fallback = img
                        break
                if not matched_fallback and all_imgs:
                    matched_fallback = all_imgs[0]
                
                if matched_fallback:
                    st.image(matched_fallback, caption=f"Preview for {current_id} (Matched provider)", use_container_width=True)
                else:
                    st.warning(f"Image not found for {current_id}")
                
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
    
    csv_path = "output/level2.csv"
    if not os.path.exists(csv_path):
        csv_path = "test/csv/level2.csv"

    if os.path.exists(csv_path):
        df_l2 = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        df_l2.columns = [str(c).strip() for c in df_l2.columns]
        for col in ['bill_id', 'question_id', 'question', 'answer']:
            if col in df_l2.columns:
                df_l2[col] = df_l2[col].astype(str).str.strip()
    else:
        df_l2 = pd.DataFrame(columns=['bill_id', 'question_id', 'question', 'answer'])

    bill_list = df_l2['bill_id'].unique().tolist() if not df_l2.empty else []

    st.sidebar.subheader("Select Q&A Bill ID")
    if bill_list:
        selected_qa_bill = st.sidebar.radio("Choose Bill for Q&A", bill_list, key="sidebar_qa_radio_new")
    else:
        selected_qa_bill = None
        st.sidebar.info("No bills found in level2.csv")

    if selected_qa_bill:
        sub_df = df_l2[df_l2['bill_id'] == selected_qa_bill].copy()
        
        def extract_num(qid):
            import re
            nums = re.findall(r'\d+', str(qid))
            return int(nums[0]) if nums else 999
        
        sub_df['q_num'] = sub_df['question_id'].apply(extract_num)
        sub_df = sub_df.sort_values('q_num')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Bill Image ({selected_qa_bill})")
            current_id = str(selected_qa_bill).strip()
            found_images = glob.glob(f"**/{current_id}.*", recursive=True)
            valid_images = [img for img in found_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if valid_images:
                st.image(valid_images[0], use_container_width=True)
            else:
                all_imgs = [img for img in glob.glob("**/*.*", recursive=True) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                matched_fallback = None
                prefix = current_id.split("_")[0].upper()
                for img in all_imgs:
                    if prefix in img.upper():
                        matched_fallback = img
                        break
                if not matched_fallback and all_imgs:
                    matched_fallback = all_imgs[0]
                
                if matched_fallback:
                    st.image(matched_fallback, caption=f"Preview for {current_id}", use_container_width=True)
                else:
                    st.warning("Image not found for this bill ID.")

        with col2:
            st.subheader(f"Q&A for {selected_qa_bill} ({len(sub_df)} Questions)")
            for _, row in sub_df.iterrows():
                qid = row.get('question_id', '')
                q_text = row.get('question', '')
                ans_text = row.get('answer', '')
                
                st.markdown(f"**Q ({qid}):** {q_text}")
                if pd.isna(ans_text) or str(ans_text).strip() == "" or str(ans_text).lower() == "nan":
                    st.markdown("*Answer: [Pending/Not Answered]*")
                else:
                    st.success(f"**Answer:** {ans_text}")
                st.markdown("---")

    st.subheader("Full Q&A Table")
    if not df_l2.empty:
        display_cols = [c for c in ['bill_id', 'question_id', 'question', 'answer'] if c in df_l2.columns]
        st.dataframe(df_l2[display_cols], use_container_width=True)
    else:
        st.warning("level2.csv is empty or missing.")

else:
    st.header("Bill Images Gallery")
    all_imgs = glob.glob("train_bytes/*.*") + glob.glob("train_bills/*.*") + glob.glob("test/bills/*.*")
    if all_imgs:
        cols = st.columns(3)
        for idx, img_path in enumerate(all_imgs):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info("No bill images found.")
