from base64 import b64encode
import streamlit as st
import uuid
from ai_doctor_agent import graph, generate_pdf
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage


# Cache the MemorySaver so it's not recreated
@st.cache_resource
def get_memory():
    return MemorySaver()


memory = get_memory()

# Initialize thread_id and state once
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "message_type": None,
        "name": None,
        "age": None,
        "gender": None,
        "history": None,
        "diagnosed": False
    }
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Medical Assistant Chat", layout="wide")
st.title("Medical Assistant")

# Sidebar: collect demographics first
st.sidebar.header("Patient Info")
name_input = st.sidebar.text_input("Full Name", value=st.session_state.state.get("name", ""))
age_input = st.sidebar.number_input("Age", min_value=0, step=1, value=st.session_state.state.get("age", 0))
gender_options = ["", "Male", "Female", "Other"]
current_gender = st.session_state.state.get("gender", "")
gender_index = gender_options.index(current_gender) if current_gender in gender_options else 0
gender_input = st.sidebar.selectbox("Gender", gender_options, index=gender_index)
history_input = st.sidebar.text_area("Medical History", value=st.session_state.state.get("history", ""))

if st.sidebar.button("Submit Info"):
    st.session_state.state["name"] = name_input
    st.session_state.state["age"] = age_input
    st.session_state.state["gender"] = gender_input
    st.session_state.state["history"] = history_input

    st.session_state.history.append({
        "role": "assistant",
        "name": "Patient Info Collector",
        "content": (
            f"Patient Info:\n"
            f"- Name: {st.session_state.state['name']}\n"
            f"- Age: {st.session_state.state['age']}\n"
            f"- Gender: {st.session_state.state['gender']}\n"
            f"- History: {st.session_state.state['history']}\n\n"
            "You may now describe your symptoms."
        )
    })

# Show chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f"**{msg['name']}:** {msg['content']}")
    elif msg["role"] == "tool":
        with st.expander(f"🔬PubMed result", expanded=False):
            st.text(f'Query: {msg["query"]}')
            st.markdown(msg["content"])

# Only enable chatting after info collected
if st.session_state.state["name"] and st.session_state.state["age"] and st.session_state.state["gender"]:
    user_prompt = st.chat_input("Describe your symptoms here…")
    if user_prompt:
        # Immediately show user's message
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(f"You:\n{user_prompt}")

        st.session_state.state["messages"].append({"role": "user", "content": user_prompt})

        with st.spinner("Assistant is thinking…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            st.session_state.state = graph.invoke(st.session_state.state, config=config)

        last = st.session_state.state["messages"][-1]

        st.session_state.history.append({
            "role": "assistant",
            "name": last.name,
            "content": last.content
        })

        with st.chat_message("assistant"):
            st.markdown(f"{last.name}:\n{last.content}")

        for m in st.session_state.state["messages"][-2:]:  # look at recent messages
            if isinstance(m, ToolMessage):

                st.session_state.history.append({
                    "role": "tool",
                    "name": m.name,
                    "content": m.content,
                    "query": m.additional_kwargs.get("query", "") if m.additional_kwargs else ""
                })

                with st.expander(f"🔬PubMed result", expanded=False):
                    st.text(f"Query: {m.additional_kwargs.get('query', '')}")
                    st.markdown(m.content)

        report_text = st.session_state.state.get("report")

        if report_text:
            if "pdf_bytes" not in st.session_state:
                st.session_state.pdf_bytes = generate_pdf(report_text)
            pdf_bytes = st.session_state.pdf_bytes

        if st.session_state.get("pdf_bytes"):
            b64 = b64encode(st.session_state.pdf_bytes).decode("utf-8")
            pdf_html = (
                f'<embed src="data:application/pdf;base64,{b64}" '
                'width="700" height="900" type="application/pdf">'
            )
            st.markdown(pdf_html, unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Medical Report (PDF)",
                data=st.session_state.pdf_bytes,
                file_name="medical_report.pdf",
                mime="application/pdf",
                key="download_report"
            )
