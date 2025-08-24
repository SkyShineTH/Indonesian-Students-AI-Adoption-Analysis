import streamlit as st

st.set_page_config(
    page_title="AI Adoption Dashboard",
    page_icon="",
    layout="wide",
    menu_items={"About": "AI Adoption among Indonesian Students – Streamlit Dashboard"}
)

st.title("Indonesian Students AI Adoption – Dashboard")
st.write(
    "สำรวจการใช้ AI ของนักศึกษาจากผลสำรวจ: Demographics, Usage Patterns, Clustering, และ Statistical Modeling.\n"
    "ใช้เมนูซ้าย (Pages) เพื่อไปยังแต่ละส่วนได้เลย"
)

st.markdown("---")
st.subheader("วิธีใช้")
st.markdown("""
1) ไปที่ **Overview** เพื่อดู Distribution, AI Tools และ Correlation  
2) ไปที่ **Clustering** เพื่อทดลองเลือก k และดูโปรไฟล์คลัสเตอร์  
3) ไปที่ **Statistical Modeling** เพื่อรัน ANOVA/Regression แบบโต้ตอบ
""")
