import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader

# Set up the Streamlit app configuration
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📺")
st.title("📺 YouTube Video Summarizer")

# Sidebar for Groq API Key
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Main section for video summarization
st.subheader("Enter YouTube Video URL")
video_url = st.text_input("YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Video"):
    # Validate the inputs
    if not groq_api_key.strip() or not video_url.strip():
        st.error("Please provide the API key and YouTube video URL.")
    elif not validators.url(video_url) or "youtube.com" not in video_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Load the YouTube video data
                loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
                docs = loader.load()

                # Summarize the content
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the summary
                st.success("Summary:")
                st.write(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")
