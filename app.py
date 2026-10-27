"""Streamlit demo.  Run:  streamlit run app.py"""
from datetime import date

import streamlit as st

from letter_generator import CATEGORY_LABELS, classify_reason, draft_letter, load_model

st.set_page_config(page_title="Leave Letter Assistant", page_icon="📝")
st.title("📝 Leave Letter Assistant")
st.caption(
    "Drafts leave applications for genuine absences. Signature lines are left blank: "
    "a parent/guardian must sign, and medical certificates must come from a real doctor."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

reason = st.text_area("Why will you be absent? (describe the real reason)", placeholder="I have a high fever and the doctor advised rest")

col1, col2 = st.columns(2)
name = col1.text_input("Your name")
class_name = col2.text_input("Class / section")
recipient = st.text_input("Addressed to", value="The Class Teacher / Head of Department")

d1, d2 = st.columns(2)
from_date = d1.date_input("From", value=date.today())
to_date = d2.date_input("To", value=date.today())

tone = st.radio("Tone", ["formal", "polite_casual"], horizontal=True)

if st.button("Generate letter", type="primary"):
    if not reason.strip() or not name.strip():
        st.warning("Please enter your name and the reason.")
    elif to_date < from_date:
        st.warning("'To' date can't be before 'From' date.")
    else:
        label, conf = classify_reason(model, reason)
        st.info(f"Detected category: **{CATEGORY_LABELS[label]}** (confidence {conf:.0%})")
        if conf < 0.4:
            st.warning("The model isn't very sure. Please check the category before using the letter.")
        letter = draft_letter(label, name, class_name or "—", recipient, from_date, to_date, reason, tone)
        st.text_area("Your draft", letter, height=420)
        st.download_button("Download as .txt", letter, file_name="leave_letter.txt")
