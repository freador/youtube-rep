
import streamlit as st
import main as la
import textwrap

st.title("Assistente do youtube")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(label="URL do Video", max_chars=56)
        query = st.sidebar.text_area(
            label="Me pergunte algo sobre o video", max_chars=500, key="query"
        )
        submit_button = st.form_submit_button(label="enviar")

if query and youtube_url:
    db = la.create_vector_from_yt_url(youtube_url)
    response, docs = la.get_response_from_query(db, query)
    st.subheader("Respostas")
    st.text(textwrap.fill(response["answer"]),width=85)
