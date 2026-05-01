import streamlit as st
import os
from dotenv import load_dotenv

from src.visualize import visualize_mind_map
from utils.generate_material import generate_study_materials
from utils.load_data import extract_text_from_pdf

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("AI-Powered Study Assistant")
st.markdown("Generate a **deep topic-only mind map** and flashcards from your notes, PDFs, or a topic name.")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.warning("GROQ_API_KEY not found. Please add it to your .env file or enter it below.")
    groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")

if 'study_guide' not in st.session_state:
    st.session_state.study_guide = None
if 'mind_map_image' not in st.session_state:
    st.session_state.mind_map_image = None

with st.sidebar:
    st.header("📚 Input Your Content")
    input_method = st.radio("Choose your input method:", ("Topic Name", "Plain Text Notes", "PDF File"))

    content_input = ""
    if input_method == "Topic Name":
        content_input = st.text_input("Enter the topic name:", placeholder="e.g., Machine Learning")
        content_input = content_input.strip()
        content_input = content_input.split(" ")
        content_input = "-".join(content_input)
    elif input_method == "Plain Text Notes":
        content_input = st.text_area("Paste your notes here:", height=300, placeholder="Paste your detailed notes...")
    elif input_method == "PDF File":
        uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                content_input = extract_text_from_pdf(uploaded_file)

    if st.button("Generate Study Materials", use_container_width=True, type="primary"):
        if not groq_api_key:
            st.error("Please provide a GROQ API Key to proceed.")
        elif not content_input or len(content_input.strip()) < 2:
            st.error("Input is too short. Please enter a valid topic, paste more detailed notes, or upload a valid PDF.")
        else:
            with st.spinner("The AI is thinking... Generating your deep topic hierarchy..."):
                try:
                    study_guide = generate_study_materials(content_input, groq_api_key)
                    if study_guide and 'mind_map' in study_guide and study_guide['mind_map']:
                        st.session_state.study_guide = study_guide
                        with st.spinner("Creating detailed mind map visualization..."):
                            mind_map_path = visualize_mind_map(study_guide['mind_map'])
                            st.session_state.mind_map_image = mind_map_path
                        st.success("Successfully generated study materials!")
                    else:
                        st.session_state.study_guide = None
                        st.session_state.mind_map_image = None
                        st.error("Generation failed. The model might not have returned the expected data. Please try again.")
                except Exception as e:
                    st.error(f"A critical error occurred: {e}")
                    st.session_state.study_guide = None
                    st.session_state.mind_map_image = None


if st.session_state.study_guide:
    st.markdown("---")
    st.header("Your Study Guide")

    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Mind Map")
        if st.session_state.mind_map_image:
            st.image(st.session_state.mind_map_image, caption="Generated Mind Map", use_container_width=True)
            with open(st.session_state.mind_map_image, "rb") as file:
                st.download_button(
                    label="Download Mind Map (PNG)",
                    data=file,
                    file_name="mind_map.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("Could not generate mind map image.")

    with col2:
        st.subheader("Flashcards")
        flashcards = st.session_state.study_guide.get('flashcards', [])
        if flashcards:
            for i, card in enumerate(flashcards):
                with st.expander(f"**Question {i+1}:** {card['question']}"):
                    st.info(f"**Answer:** {card['answer']}")
        else:
            st.info("No flashcards were generated.")

    # with st.expander("View Raw Mind Map Data (JSON)"):
    #     st.json(st.session_state.study_guide.get('mind_map', {}))

else:
    st.info("Your generated study materials will appear here once you provide an input and click 'Generate'.")

