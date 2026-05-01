from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from utils.structure import StudyGuide

load_dotenv()

def generate_study_materials(content: str, groq_api_key: str) -> dict:
    """
    Generates a mind map and flashcards using LangChain and ChatGroq.
    """
    try:
        model = ChatGroq(
            temperature=0,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key
        )

        parser = JsonOutputParser(pydantic_object=StudyGuide)

        prompt = PromptTemplate(
            template="""
            You are a world-class AI expert at creating deeply hierarchical outlines.
            Your task is to analyze the provided text and create a comprehensive study guide in a structured JSON format.
            STRICT INSTRUCTION:
            - You must ONLY respond if the provided content is educational, academic, or study-related.
            - If the content is NOT study-related (e.g., casual conversation, jokes, personal queries, entertainment, etc.), you MUST NOT generate the study guide.
            - In such cases, return ONLY this JSON:

            The study guide must contain two main keys: "mind_map" and "flashcards".

            1.  **mind_map**:
                -   This must be a deeply nested JSON object representing a pure topic-subtopic hierarchy.
                -   **Keys** must always be the name of a topic or concept.
                -   **Values** must ALWAYS be another nested JSON object representing the sub-topics for that key.
                -   **CRITICAL RULE: Do NOT include any descriptions, definitions, or explanations as string values.** The mind map must ONLY contain the structure of topics. If a topic has no further sub-topics, its value should be an empty object {{}}.
                -   Your goal is to create the deepest and most exhaustive hierarchy possible, breaking down every concept from the source text into its constituent parts.

            2.  **flashcards**:
                -   This must be a list of 10 insightful flashcards based on the most important information in the text.
                -   Each flashcard should be a JSON object with a "question" and an "answer" field.

            Please process the following content and generate the detailed study guide.

            IMPORTANT RULES:
            - Output must be strictly valid JSON.
            - Do not include any extra text outside JSON.
            - Do not explain your reasoning.
            - Do not generate partial outputs.
            - Ensure high-quality, exam-focused content.

            CONTENT:
            {content}

            FORMAT INSTRUCTIONS:
            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        response = chain.invoke({"content": content})
        return response

    except Exception as e:
        return e