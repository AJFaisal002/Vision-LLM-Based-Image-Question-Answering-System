import os
import pathlib
import tempfile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

# -------------------------------------------------
# Streamlit page config (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------
st.set_page_config(
    page_title="Vision-LLM Image QA",
    layout="centered"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #fafafa;
        }
        .title-text {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle-text {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }
        footer {
            visibility: hidden;
        }
        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #666;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            border-top: 1px solid #eaeaea;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Initialize LangChain Agent
# --------------------------------
tools = [
    ImageCaptionTool(),
    ObjectDetectionTool()
]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)

# --------------------------------
# Streamlit UI
# --------------------------------
st.markdown(
    '<div class="title-text">🖼️ Ask a Question About an Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'Upload an image and ask natural language questions powered by Vision + LLMs'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)

    user_question = st.text_input(
        "Ask a question about the image:"
    )

    if user_question:
        suffix = pathlib.Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            image_path = tmp.name

        try:
            with st.spinner("Analyzing image..."):
                response = agent.run(
                    f"{user_question}, this is the image path: {image_path}"
                )
                st.success("Answer:")
                st.write(response)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

# --------------------------------
# Footer
# --------------------------------
st.markdown(
    """
    <div class="custom-footer">
        © 2026 Adnan Faisal. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
