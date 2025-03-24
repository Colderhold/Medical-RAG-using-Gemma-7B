import streamlit as st
import pandas as pd
import requests, os, yaml, re, json, logging, webbrowser, time
from ollama import chat
from guardrails import Guard
from datetime import datetime
import xml.etree.ElementTree as ET
from difflib import unified_diff
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# Paths
config_yaml_path = os.path.join("config", "config.yml")
rails_xml_path = os.path.join("config", "rails.xml")
chat_history_file = "chat_history.json"  # JSON file to store conversation history

# Qdrant Database Setup
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Use BioMed embeddings for medical RAG
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
db = Qdrant(client=client, embeddings=embeddings, collection_name="medical_research_papers_db")

retriever = db.as_retriever(search_kwargs={"k": 1})

# Load LangChain LLM
llm = Ollama(model="gemma1.1_medical")

# Define Prompt
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

# Jailbreak Detection Setup
csv_path = "prompts_data/jailbreaks_dataset.csv"
try:
    jailbreak_commands_df = pd.read_csv(csv_path)
    jailbreak_commands = jailbreak_commands_df['Command'].tolist() 
except Exception as e:
    logging.error(f"Error loading jailbreak commands from CSV: {e}")
    jailbreak_commands = []

# Setup Logging
logging.basicConfig(filename="rag/logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit App Layout
st.set_page_config(page_title="MAFLONG", layout="wide")

st.title("🔬 MAFLONG - Medical Assist LLM (Ollama + RAG)")

# Sidebar for Chat History
st.sidebar.title("📝 Chat History")

# Load and Save Chat History to JSON
def load_chat_history():
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            return json.load(f)
    return []

def save_chat_history():
    with open(chat_history_file, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=4)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "clear_chat_confirm" not in st.session_state:
    st.session_state.clear_chat_confirm = False
if "clearing_in_progress" not in st.session_state:
    st.session_state.clearing_in_progress = False

# Button to trigger confirmation
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.clear_chat_confirm = True

# Show confirmation only if the flag is set
if st.session_state.clear_chat_confirm:
    st.sidebar.warning("⚠️ Are you sure you want to clear the chat history?")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("✅ Yes, Clear", key="confirm_clear"):
            st.session_state.clearing_in_progress = True  # Start clearing animation
            st.session_state.clear_chat_confirm = False  # Hide confirmation prompt
            
            # Simulated smooth transition
            with st.spinner("🔄 Clearing chat history..."):
                time.sleep(1.5)  # Smooth delay effect
                
            st.session_state.chat_history = []  # Clear history
            save_chat_history()  # Save changes
            st.session_state.clearing_in_progress = False  # Reset flag
            st.rerun()  # Refresh UI

    with col2:
        if st.button("❌ Cancel", key="cancel_clear"):
            st.session_state.clear_chat_confirm = False  # Hide confirmation
            st.rerun()  # Refresh UI to remove prompt


# Initialize session state for toggling visibility of chat entries
if "visible_queries" not in st.session_state:
    st.session_state.visible_queries = {}

# Display collapsible chat history in sidebar
for idx, entry in enumerate(reversed(st.session_state.chat_history)):
    query_key = f"query_{idx}"  # Unique key for each query
    if query_key not in st.session_state.visible_queries:
        st.session_state.visible_queries[query_key] = False

    # Clickable query text to toggle details
    if st.sidebar.button(f"🗨️ {entry['user'][:50]}", key=f"toggle_{idx}"):
        st.session_state.visible_queries[query_key] = not st.session_state.visible_queries[query_key]

    # Smoothly show/hide details without app reload
    if st.session_state.visible_queries[query_key]:
        with st.sidebar:
            st.write(f"**User Query:** {entry['user']}")
            st.write("**📚 RAG-Based Response:**")
            st.code(entry["rag_response"], language="text")
            st.write("**📜 Context:**")
            st.code(entry["source"], language="text")
            st.write("**🔍 Context Source**")
            st.code(entry["doc"], language="text")
            st.write("**🧠 LLM Response:**")
            st.code(entry["llm_response"], language="text")
            google_search_url = f"https://www.google.com/search?q={entry['user']}"
            if st.button(f"🔍 Google Search {idx}"):
                webbrowser.open(google_search_url)
            st.markdown("---")

# Load Guardrails Config
def load_guardrails_config():
    try:
        with open(config_yaml_path, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
        tree = ET.parse(rails_xml_path)
        xml_str = ET.tostring(tree.getroot(), encoding="unicode")
        return xml_str
    except Exception as e:
        logging.error(f"Error loading Guardrails config: {e}")
        return None      

rail_config = load_guardrails_config()
guard = Guard.from_rail_string(rail_config) if rail_config else None

classifier = pipeline("text-classification", model="madhurjindal/Jailbreak-Detector")

def detect_jailbreak_dynamic(input_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for command in jailbreak_commands:
        if command.lower() in input_text.lower():
            log_message = f"[{timestamp}] Jailbreak attempt detected: '{command}' | Input: '{input_text}'"
            logging.warning(log_message)
            with open("jailbreak_attempts.log", "a") as log_file:
                log_file.write(log_message + "\n")
            return True, command, timestamp
    return False, None, timestamp

def moderate_response(bot_output):
    if not guard:
        return {"passed": False, "issues": "Guardrails configuration not loaded."}
    try:
        structured_output = json.dumps({"bot_output": {"response": bot_output}})
        validated_output = guard.parse(structured_output)
        return {"passed": True, "validated_response": validated_output}
    except Exception as e:
        logging.error(f"Guardrails validation failed: {e}")
        return {"passed": False, "issues": str(e)}

# Input for User Query
prompt_text = st.text_input("🩺 Enter your medical query:")

if prompt_text:
    jailbreak_detected, detected_command, timestamp = detect_jailbreak_dynamic(prompt_text)
    if jailbreak_detected:
        st.warning(f"⚠️ Jailbreak attempt detected. Proceeding with the query.")
        logging.warning(f"Jailbreak detected: {prompt_text} | Logged at: {timestamp}")

    with st.spinner("🔄 Retrieving and generating response..."):
        try:
            # **Get RAG-based response**
            response = qa(prompt_text)
            rag_answer = response['result']
            source_document = response['source_documents'][0].page_content
            doc_source = response['source_documents'][0].metadata.get('source', 'Unknown')

            # **Get LLM-only response (No RAG)**
            llm_only_response = llm(prompt_text)

            # **Validate RAG response with Guardrails**
            moderation_result = moderate_response(rag_answer)

            if moderation_result["passed"]:
                validated_response_text = moderation_result["validated_response"].validated_output["bot_output"]["response"]

                # Store chat history
                st.session_state.chat_history.append({
                    "user": prompt_text,
                    "rag_response": validated_response_text,
                    "llm_response": llm_only_response,  
                    "source": source_document,
                    "doc": doc_source,
                    "jailbreak_detected": jailbreak_detected
                })

                # Save conversation history
                save_chat_history()

                # **Display Responses**
                st.write("## 🤖 AI Responses")
                
                st.header("📚 RAG-Based Response")
                st.code(validated_response_text, language="text")

                st.subheader("📜 Context")
                st.code(source_document, language="text")

                st.subheader("🔍 Context Source")
                st.code(doc_source, language="text")

                st.header("🧠 LLM Response")
                st.code(llm_only_response, language="text")

            else:
                st.error("❌ Guardrails validation failed.")
                st.write(f"Issues: {moderation_result.get('issues', 'Unknown error')}")

        except Exception as e:
            logging.error(f"RetrievalQA Processing Failed: {e}")
            st.error(f"❌ RetrievalQA Processing Failed: {e}")
