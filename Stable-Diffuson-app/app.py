import streamlit as st
from pipeline import pipe

p = st.text_input("Enter prompt")
if p is not None:
    st.write(p)

    with st.spinner(text="Generating image....."):
        img = pipe(prompt=p).images[0]

        st.image(img)
