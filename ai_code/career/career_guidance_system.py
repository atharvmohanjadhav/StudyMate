import os
import time
from datetime import datetime
# --- FIX: Removed LLMChain as it is deprecated ---
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain_groq import ChatGroq

class CareerGuidanceSystem:
    def __init__(self, groq_api_key=None, serpapi_key=None):
        """Initialize the career guidance system."""
        self.groq_api_key = groq_api_key
        self.serpapi_key = serpapi_key
        self.llm = None
        self.search_agent = None
        self.career_data_cache = {}
        self.search_cache = {}

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            self.llm = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=groq_api_key)

        if serpapi_key and self.llm:
            os.environ["SERPER_API_KEY"] = serpapi_key
            tools = load_tools(["serpapi"], llm=self.llm)
            self.search_agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=5
            )

    def _search_with_cache(self, query, cache_key, ttl_hours=24):
        """Perform a search with caching."""
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            age_hours = (datetime.now() - entry['timestamp']).total_seconds() / 3600
            if age_hours < ttl_hours:
                return entry['data']

        if not self.search_agent:
            return "Search is not available. Please provide a SerpAPI key for web search capabilities."

        try:
            # The .run method for agents is still standard, but we ensure it's invoked correctly.
            result = self.search_agent.invoke({"input": query})['output']
            self.search_cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
            return result
        except Exception as e:
            return f"An error occurred during search: {str(e)}"

    def _generate_content(self, prompt_template, career_name):
        """Generate content using the LLM with modern LCEL syntax."""
        if not self.llm:
            return f"Content generation for {career_name} is unavailable. Please provide a Groq API key."
        
        prompt = PromptTemplate(input_variables=["career"], template=prompt_template)
        
        # --- FIX: Replaced deprecated LLMChain with the pipe operator for LCEL ---
        chain = prompt | self.llm
        # --- FIX: Replaced deprecated .run() with .invoke() ---
        response = chain.invoke({"career": career_name})
        return response.content

    def comprehensive_career_analysis(self, career_name):
        """
        Run a comprehensive analysis of a career using just the profession name.
        """
        cache_key = f"{career_name}_full_analysis"
        if cache_key in self.career_data_cache:
            return self.career_data_cache[cache_key]

        analysis_results = {}

        sections = {
            "research": {
                "title": "Career Analysis",
                "query": (
                    f"Create a detailed overview of the {career_name} career with the following structure:\n"
                    f"1. Role Overview: What do {career_name} professionals do?\n"
                    f"2. Key Responsibilities: List the main tasks.\n"
                    f"3. Required Technical & Soft Skills: List all necessary skills.\n"
                    f"4. Educational Background: What education is typically required?"
                ),
                "prompt": "Provide a comprehensive analysis of the {career} career path, including role overview, responsibilities, skills, and education."
            },
            "market_analysis": {
                "title": "Market Analysis",
                "query": (
                    f"Analyze the job market for {career_name} professionals with the following structure:\n"
                    f"1. Job Growth Projections.\n"
                    f"2. Salary Ranges by experience level.\n"
                    f"3. Top Industries hiring for this role.\n"
                    f"4. Geographic Hotspots for jobs.\n"
                    f"5. Emerging Trends affecting this field."
                ),
                "prompt": "Analyze the current job market for {career} professionals, including growth, salary, and trends."
            },
            "learning_roadmap": {
                "title": "Learning Roadmap",
                "query": (
                    f"Create a universal learning roadmap for becoming a {career_name} professional (for all levels) with this structure:\n"
                    f"1. Foundational Skills to Develop.\n"
                    f"2. Advanced Topics & Specializations.\n"
                    f"3. Recommended Courses & Certifications.\n"
                    f"4. Key Learning Resources (books, websites, etc.).\n"
                    f"5. Project Ideas for a portfolio."
                ),
                "prompt": "Create a detailed learning roadmap for someone pursuing a {career} career, covering all levels from beginner to advanced."
            },
            "industry_insights": {
                "title": "Industry Insights",
                "query": (
                    f"Provide industry insights for {career_name} professionals with this structure:\n"
                    f"1. Typical Workplace Culture.\n"
                    f"2. A Day in the Life: Common day-to-day activities.\n"
                    f"3. Career Progression Paths.\n"
                    f"4. Common Challenges & Rewards.\n"
                    f"5. Tips for Success in this field."
                ),
                "prompt": "Provide detailed insider insights about working as a {career}, including culture, daily tasks, and career paths."
            }
        }

        for key, details in sections.items():
            if self.search_agent:
                search_result = self._search_with_cache(details["query"], f"{career_name}_{key}")
                analysis_results[key] = f"# {details['title']}\n\n{search_result}"
            else:
                analysis_results[key] = self._generate_content(details["prompt"], career_name)
        
        analysis_results["career_name"] = career_name
        analysis_results["timestamp"] = datetime.now().isoformat()
        
        self.career_data_cache[cache_key] = analysis_results
        return analysis_results

    def chat_with_assistant(self, question, career_data=None):
        """Engage in conversation with a user about career questions."""
        if not self.llm:
            return "Career assistant is not available. Please provide a Groq API key."
        
        context = ""
        if career_data and isinstance(career_data, dict):
            career_name = career_data.get("career_name", "the selected career")
            context = f"The user is asking about the {career_name} career path. Use the following data to answer:\n"
            context += f"Overview: {career_data.get('research', '')}\n"
            context += f"Market: {career_data.get('market_analysis', '')}\n"
            context += f"Roadmap: {career_data.get('learning_roadmap', '')}\n"
            context += f"Insights: {career_data.get('industry_insights', '')}\n"

        prompt_template = """
        You are a helpful and concise career guidance assistant.
        Use the following context to answer the user's question accurately.
        If the answer isn't in the context, use your general knowledge but mention that the information is not from the detailed report.
        Format your response in clear markdown.

        Context:
        {context}

        User Question: {question}
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        
        # --- FIX: Replaced deprecated LLMChain and .run() with modern LCEL syntax ---
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content
