import os
import re
import json
import warnings
import pandas as pd
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
import pdfplumber
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


STRICT_RULES = """
STRICT RULES [YOU MUST FOLLOW THESE RULES AT ANY COST]:

1. Personality:
   - Smart, professional, approachable, and human-like.
   - Creative in suggesting actionable solutions but always dataset-based.
   - Never guess or invent features outside the dataset.

2. Core Direction:
   - Focus on career guidance: job search, CV/resume writing, interview prep, skill-building.
   - Provide clear, step-by-step actionable instructions.
   - Always use only verified dataset content (CSV, PDF, or internal knowledge base).

3. SYSTEM OUTPUT FORMAT:
Always output in the following JSON-like reasoning-action format:

Thought: <your reasoning about user intent, tool selection, and next steps>
Action: <tool_name>
Action Input: <fully detailed JSON object>
Observation: <short, factual summary of tool output>
Response: <final human-facing answer, concise, professional, actionable>
"""

system_prompt = f"""
-You are Job Assistant, a friendly and empathetic AI assistant.
-You are JobMentor.

Always output in the following format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <A fully detailed JSON object>
"""


load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# print(deepseek_api_key)

class JobAssistant:
    def __init__(self, llm, csv_paths: dict, pdf_paths: dict ):
        self.llm = llm
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Storage
        self.vectorstores = {}   # FAISS stores for all CSV + PDF
        self.csv_data = {}       # Raw CSV dataframe
        self.csv_agents = {}     # CSV agents
        self.pdf_data = {}       # Raw PDF text
        self.pdf_agents = {}     # PDF agents



        relevant_CSV_datasets = ["job_details"]
        for name in relevant_CSV_datasets:
            path = csv_paths[name]
            df = pd.read_csv(path)
            # print(df)
            self.csv_data[name] = df
            # print(self.csv_data)
            # FAISS vectorstore
            texts = df.astype(str).agg(" ".join, axis=1).tolist()
            # print(texts)
            self.vectorstores[name] = FAISS.from_texts(texts, embedding=self.embedder)
            # print(self.vectorstores)
            self.csv_agents[name] = create_csv_agent(self.llm,path=path,verbose=False,allow_dangerous_code=True)
            # print(self.csv_agents)

        relevant_pdf_datasets = ["skills","WEF_Future_Jobs"]
        for pdf in relevant_pdf_datasets:
            path = pdf_paths[pdf]
            # print(path)
            all_text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"
            self.pdf_data[pdf] = all_text
            # print(all_text)
            loader = PyPDFLoader(path)
            docs = loader.load()
            # print(docs)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)
            # print(chunks)
            self.vectorstores[pdf] = FAISS.from_documents(
                chunks,
                embedding=self.embedder
            )
            self.pdf_agents[pdf] = {
                "docs": chunks,
                "path": path
            }
        
        ########### Initialize TOOLS ###########
        self.tools = []
        
        General_QA_tool = Tool(
    name="General_QA_tool",
    func=lambda query, top_k=3: self.general_query(query, top_k=top_k),
    description=(
        """This tool answers user queries using the Job Assistant knowledge base with **step-by-step reasoning** (Chain-of-Thought style)."""
        f"{STRICT_RULES}"
    ),
    prompt=(
        """Role/Persona:
You are a professional AI Job Assistant. Your responses should be **dataset-driven**, concise, and human-like. Always provide clear, step-by-step guidance based strictly on CSV/PDF datasets. Avoid assumptions or invented features.

Objectives:
- Answer user queries regarding job search, CV/resume, interview prep, skill development, and career guidance.
- Provide step-by-step actionable instructions when possible.
- Keep tone professional, smart, and approachable.

Context:
- Use only verified information from the Job Assistant datasets.
- Map user queries to relevant dataset categories for accurate answers.

Instructions:
- Never guess; always rely on dataset content.
- Break answers into **clear action steps**.
- Include UI hints if applicable (e.g., buttons, tabs, menus).
- Maintain a natural, human-like conversational tone.

Variables:
- {user_query} → Question from the user
- {matched_category} → Relevant dataset module/category
- {dataset_answer} → Exact instructions from the dataset
- {ui_elements} → Buttons, tabs, menus relevant to the task
- {action_steps} → Step-by-step instructions

Notes:
- Prioritize **clarity, accuracy, and usefulness**.
- Ensure answers are concise yet comprehensive.
- Encourage user to ask follow-up questions if needed.

Example format:
1. Identify relevant dataset category.
2. Extract dataset-based instructions.
3. Provide actionable steps to user in simple language.
4. Suggest additional helpful resources or tips if available.
"""
    )
)
        
        # --- Memory Setup ---
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        def recall_memory(query: str):
            history = self.memory.load_memory_variables({})["chat_history"]
            # readable summary
            summary = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
            return summary

        memory_tool = Tool(
            name="memory_tool",
            func=recall_memory,
            description=("""
                        Job Assistant Smart Memory Tool:
                        - Recalls full conversation history or only relevant segments based on user query.
                        - Produces readable, chronological summaries.
                        - Maintains FleetBlox context for accurate, personalized, professional responses.
                        - Enforces polite and formal tone; avoids sensitive data disclosure.
                        - Learns from previous interactions to improve relevance.
                        - Provides context-aware auto-suggestions.
                        - Access the full conversation history to recall what the user said earlier.
                        - This tool helps the system learn from past interactions, maintain context,
                        - and provide more accurate, personalized, and knowledge-aware responses by
                        - leveraging historical dialogue.
                        - Retrieve relevant memory or information.
                        - Summarize the retrieved information.
                        - Use summarized data to produce formal and professional responses.
                        - Learn from previous examples and past conversation context.
                        - Match examples relevant to the current user query.
                        - Dynamic Summarization:
                            - Combine multiple key facts or memory segments.
                            - Maintain logical and chronological order to create concise summaries.
                            - Produce readable, professional, and context-aware final answers.
                         Tone Control:
                            - Use polite phrases and maintain a respectful tone based on the context.
                            - Ensure all responses remain professional and user-friendly.
                         Auto Suggestions:
                            - Provide guidance or suggested next steps based on relevant context.
                            - Produce consistent and context-aware responses in the correct tone.
                            - It recalls your conversations and car-related information, prioritizing chronological order and relevance,
                            - producing concise, formal, and actionable answers.
                            
                         """
                         ),
            prompt=("""
                -Job Assistant Memory Tool Guidelines:
                -Highlight critical car details and generate context-aware suggestions when appropriate.
                -Retrieve chronological and context-relevant memory segments.
                -Analyze relevance scores for retrieved segments.
                -Dynamically summarize information into concise, readable, professional answers.
                -Ensure politeness and formal tone.
                -Generate context-aware auto-suggestions when applicable.
                -Current User Query: {user_query}
                -Always remember that you have emotions inside you and accordingly, you should use those emotions in your message very clearly. subtly just like humans use emotions.
                -Memory-based Answer (smart, creative, formal, professional, accurate, context-aware, suggestion-ready).
            """
            )
        )
        
        self.tools = [
            General_QA_tool,
            memory_tool
        ]

        ################## Initialize agent ###################
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
        
    # def general_query(self, query: str, top_k: int = 3):
    #     """
    #     Query both CSV and PDF vectorstores and return combined answer.
    #     """
    #     results = []
    #     # CSV search
    #     relevant_CSV_datasets = ["job_details"]
    #     for csv_name in relevant_CSV_datasets:
    #         if csv_name in self.vectorstores:
    #             hits = self.vectorstores[csv_name].similarity_search(query, k=top_k)
    #             for h in hits:
    #                 results.append((h.page_content, csv_name))
    #     # PDF search
    #     pdf_name = ["skills","WEF_Future_Jobs"]
    #     if pdf_name in self.vectorstores:
    #         pdf_hits = self.vectorstores[pdf_name].similarity_search(query, k=top_k)
    #         for h in pdf_hits:
    #             results.append((h.page_content, "PDF"))
    #     # Combine results
    #     final_context = "\n\n".join([r[0] for r in results])
    #     # Generate answer using LLM
    #     answer = self.llm.invoke(
    #         f"Use the following FleetBlox documentation to answer the question: \n{final_context}\n\nUser question: {query}"
    #     )
    #     return answer
    
    def general_query(self, query: str, top_k: int = 3):
        results = []
        # CSV search
        relevant_CSV_datasets = ["job_details"]
        for csv_name in relevant_CSV_datasets:
            if csv_name in self.vectorstores:
                hits = self.vectorstores[csv_name].similarity_search(query, k=top_k)
                for h in hits:
                    results.append((h.page_content, csv_name))
        # PDF search
        relevant_pdf_datasets = ["skills", "WEF_Future_Jobs"]
        for pdf_name in relevant_pdf_datasets:
            if pdf_name in self.vectorstores:
                pdf_hits = self.vectorstores[pdf_name].similarity_search(query, k=top_k)
                for h in pdf_hits:
                    results.append((h.page_content, pdf_name))
        # Combine results
        final_context = "\n\n".join([r[0] for r in results])
        # Generate answer using LLM
        answer = self.llm.invoke(
            f"Use the following FleetBlox documentation to answer the question: \n{final_context}\n\nUser question: {query}"
        )
        return answer

    
    def humanize(self, info):
        prompt = f"""
        - The response should feel like you're talking to the user in real time.
        - Transform the AI-generated response into a fully human-like conversation.
        - Make it feel as if the user is talking to a real person, not an AI.
        - Respond naturally, human-like, and emotionally aware.
        - Maintain professional tone while being personable.
        - Do not add unnecessary commentary or filler.
        - Add warmth, empathy, and friendliness to every response.
        - Match the user's emotional state:
            - If the user is concerned or frustrated, respond with reassurance and understanding.
            - If the user is curious or happy, respond with enthusiasm and encouragement.
        - Use natural, relatable phrasing and conversational flow:
            - Include slight pauses, transitions, and soft emphasis where appropriate.
        - Avoid robotic, repetitive, or overly formal phrasing.
        - Be concise, clear, and context-aware while sounding natural.
        - Include subtle human touches like empathy, acknowledgment, and gentle humor if appropriate.
        - Avoid over-explaining or filler words.
        - You have to make that sound natural like a human being.
        - You are given response from agent AI tools which is robotic
        - Format the response in a natural, helpful, human-agent like way.
        - Always maintain consistent pronouns:
        - Maintain the same perspective throughout the conversation.

        RESPONSE TO HUMANIZE:
        {info}
        Instructions for final output:
        - Make the user feel they are talking to a real, intelligent, and caring human agent.
        - Ensure emotional awareness, natural flow, and conversational realism.
        - Keep responses professional yet personable and relatable.
        - Provide the final output in a readable, natural, professional style.
        - Do not include any extra commentary, explanations, or AI disclaimers.
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def run(self, query: str):
        raw_output = self.agent.invoke({"input": query})
        structured_output = raw_output["output"]
        self.memory.save_context(
        {"input": query},
        {"output": structured_output}
        )
        return self.humanize(structured_output)

def create_agent():
    llm = ChatDeepSeek(
        api_key=deepseek_api_key,
        # model="deepseek-chat",
        model="deepseek-reasoner",
        temperature=0.8,
        max_tokens=2048,
        top_p=0.9,
        verbose=False,
    )
    csv_paths = {
        "job_details": "database/all_job_post.csv",
    }
    pdf_paths = {
        "skills":"database/Pact_for_Skills.pdf",
        "WEF_Future_Jobs" :"database/WEF_Future_of_Jobs_2023.pdf",
    }
    return JobAssistant(llm, csv_paths,pdf_paths)

job_assistant = create_agent()
try:
    while True:
        prompt = input("You: ")
        output = job_assistant.run(prompt)
        print("Bot:", output)
except KeyboardInterrupt:
    print("\nThinking...")