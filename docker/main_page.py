import streamlit as st





st.title(' **Welcome in our application 👋**')



st.write(
    """
    Our application consists in driving the webcam of a computer to detect the people present in real time.
    The application recognizes people who are part of the **SISE 2022-2023** class and indicates "unknown" otherwise. 
    It also indicates the emotion of the person (joy, anger, sadness, etc.), his gender and age. 
    The application controls vocally via the computer's microphone and records a video (MP4).
    **Challenge Web_Mining 2 days**
    """
)

c1, c2 = st.columns(2)
with c1:
    st.info('**Drive: [@DHH](https://drive.google.com/drive/folders/1ywsqcv9iqMzvMwoVi0ZeYtWW4mpltjZB?usp=sharing)**', icon="💡")
with c2:
    st.info('**GitHub: [@DHH](https://github.com/PierreDubrulle/web_mining.git)**', icon="💻")






st.sidebar.markdown("# Home 🎈")