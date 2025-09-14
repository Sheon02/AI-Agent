import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import logging
from dotenv import load_dotenv
from chatbot.agent import DataAnalystAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize the agent
@st.cache_resource
def get_agent():
    logger.info("Initializing DataAnalystAgent")
    agent = DataAnalystAgent()
    logger.info("DataAnalystAgent initialized successfully")
    return agent

agent = get_agent()

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide"
)
logger.info("Streamlit page configured")

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #1f2937;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
}
.stDataFrame {
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)
logger.info("Custom CSS applied")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized messages session state")

if "messages" in st.session_state:
    updated_messages = []
    for message in st.session_state.messages:
        if isinstance(message, dict):
            if "type" in message:
                # Already new format
                updated_messages.append(message)
            else:
                # Convert old format to new format
                updated_messages.append({
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                    "type": "text"
                })
        else:
            # Handle any other format (shouldn't happen)
            updated_messages.append({
                "role": "assistant",
                "content": str(message),
                "type": "text"
            })
    st.session_state.messages = updated_messages
    logger.info("Message format migration completed")
    
if "df" not in st.session_state:
    st.session_state.df = None
    logger.info("Initialized df session state")

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
    logger.info("Initialized processed_df session state")

# Sidebar for file upload
with st.sidebar:
    st.title("📊 Data Analyst Assistant")
    st.write("Upload your CSV file to get started")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            logger.info(f"File uploaded: {uploaded_file.name}")
            st.session_state.df = pd.read_csv(uploaded_file)
            logger.info(f"DataFrame loaded with shape: {st.session_state.df.shape}")
            
            st.session_state.processed_df = st.session_state.df.copy()
            logger.info("Created processed_df copy")
            
            st.success("✅ File uploaded successfully!")
            logger.info("File upload success message displayed")
            
            # Show basic info
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
            st.write("**Columns:**")
            for col in st.session_state.df.columns:
                st.write(f"• {col}")
            
            # Set the current dataframe in the agent
            agent.set_current_dataframe(st.session_state.processed_df)
            logger.info("DataFrame set in agent")
                
        except Exception as e:
            error_msg = f"Error reading file: {e}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
    
    st.markdown("---")
    st.write("**💡 Available Commands:**")
    st.write("- `Clean the data` - Remove duplicates and handle missing values")
    st.write("- `Show first 5 rows` - Display data preview")
    st.write("- `Create histogram for [column]` - Visualize distribution")
    st.write("- `Scatter plot of [x] vs [y]` - Compare two variables")
    st.write("- `Describe data` - Show statistics")
    st.write("- `Dataset info` - Show columns and data types")
    st.write("- `Handle missing values` - Clean missing data")
    st.write("- `Filter where [condition]` - Filter data")
    st.write("- `What columns are available?` - List all columns")
    st.write("- `Help` - Show available commands")

# Main chat interface
st.title("🤖 AI Data Analyst Assistant")
st.write("Chat with your data analyst assistant. Upload a CSV file and start asking questions!")

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    logger.debug(f"Displaying message {i}: {message.get('type', 'unknown')}")
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
            
            # Show dataframe preview if it exists
            if "dataframe_preview" in message:
                st.write("**Data Preview:**")
                st.dataframe(message["dataframe_preview"], use_container_width=True)
                logger.debug("Displayed dataframe preview")
            
            # Show image if it exists
            if "image_content" in message:
                st.image(message["image_content"])
                logger.debug("Displayed image content")
                
        elif message["type"] == "image":
            st.image(message["content"])
            if "title" in message:
                st.caption(message["title"])
            logger.debug("Displayed image")
        elif message["type"] == "dataframe":
            if "title" in message:
                st.write(f"**{message['title']}**")
            st.dataframe(message["content"], use_container_width=True)
            logger.debug("Displayed dataframe")

# Chat input
if prompt := st.chat_input("Ask something about your data..."):
    logger.info(f"User input received: {prompt}")
    
    if st.session_state.df is None:
        error_msg = "No CSV file uploaded"
        logger.warning(error_msg)
        st.error("Please upload a CSV file first!")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "type": "text"
    })
    logger.info("User message added to chat history")
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the query
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                logger.info(f"Processing query: {prompt}")
                
                # Set the current dataframe in the agent
                agent.set_current_dataframe(st.session_state.processed_df)
                logger.debug("Current dataframe set in agent")
                
                response = agent.process_query(
                    prompt, 
                    st.session_state.processed_df
                )
                logger.info(f"Agent response type: {response['type']}")
                
                # Handle different response types
                if response["type"] == "text":
                    st.markdown(response["content"])
                    chat_message = {
                        "role": "assistant",
                        "content": response["content"],
                        "type": "text"
                    }
                    logger.debug("Text response handled")
                
                elif response["type"] == "dataframe":
                    if "title" in response:
                        st.write(f"**{response['title']}**")
                    st.dataframe(response["content"], use_container_width=True)
                    chat_message = {
                        "role": "assistant",
                        "content": response["content"],
                        "type": "dataframe"
                    }
                    if "title" in response:
                        chat_message["title"] = response["title"]
                    logger.debug("Dataframe response handled")
                
                elif response["type"] == "image":
                    st.image(response["content"])
                    if "title" in response:
                        st.caption(response["title"])
                    chat_message = {
                        "role": "assistant",
                        "content": response["content"],
                        "type": "image"
                    }
                    if "title" in response:
                        chat_message["title"] = response["title"]
                    logger.debug("Image response handled")

                elif response["type"] == "dashboard":
                    with st.chat_message("assistant"):
                        st.image(response["content"], caption=response.get("title", "Dashboard"))
                        if "summary" in response:
                            with st.expander("Dashboard Summary"):
                                st.write(response["summary"])
                    chat_message = {"role": "assistant", "content": "📊 Dashboard generated"}
                    st.session_state.messages.append(chat_message)

                elif response["type"] == "report":
                    # Display the report content
                    st.markdown(response["content"])
                    # Display visualizations if they exist
                    if "visualizations" in response:
                        for viz_title, viz_content in response["visualizations"]:
                            st.subheader(viz_title)
                            st.image(viz_content)
                    
                    # Add message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"📊 Report generated: {response['content'][:100]}..."  # First 100 chars
                    })

                    chat_message = {"role": "assistant", "content": f"📊 Report generated: {response['content'][:100]}..."}
                    st.session_state.messages.append(chat_message)
    
                else:
                    # Handle unknown response types
                    with st.chat_message("assistant"):
                        st.write("Unknown response type from agent")
                    st.session_state.messages.append({"role": "assistant", "content": "Unknown response type"})
                
                # Handle combined results with multiple content types
                if "dataframe_preview" in response:
                    st.write("**Data Preview:**")
                    st.dataframe(response["dataframe_preview"], use_container_width=True)
                    chat_message["dataframe_preview"] = response["dataframe_preview"]
                    logger.debug("Dataframe preview added to response")

                if "image_content" in response:
                    st.image(response["image_content"])
                    chat_message["image_content"] = response["image_content"]
                    logger.debug("Image content added to response")
                
                # Add to chat history
                st.session_state.messages.append(chat_message)
                logger.info("Assistant response added to chat history")
                
                # Update processed dataframe if it was modified
                if "df" in response:
                    st.session_state.processed_df = response["df"]
                    logger.info(f"Updated processed_df with new shape: {st.session_state.processed_df.shape}")
                    # Update agent with new dataframe
                    agent.set_current_dataframe(st.session_state.processed_df)
                    logger.debug("Agent updated with new dataframe")
                    
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(error_msg, exc_info=True)
                error_display = f"❌ Error processing your request: {str(e)}"
                st.error(error_display)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_display,
                    "type": "text"
                })

# Sidebar buttons
st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Chat History"):
    logger.info("Clear chat history button clicked")
    st.session_state.messages = []
    logger.info("Chat history cleared")
    st.rerun()

if st.sidebar.button("🔄 Reset to Original Data"):
    logger.info("Reset data button clicked")
    if st.session_state.df is not None:
        st.session_state.processed_df = st.session_state.df.copy()
        logger.info("Data reset to original")
        # Update agent with reset dataframe
        agent.set_current_dataframe(st.session_state.processed_df)
        logger.debug("Agent updated with reset dataframe")
        st.success("✅ Data reset to original!")
        st.rerun()

# Display current dataset info in sidebar
if st.session_state.processed_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current Dataset:** {st.session_state.processed_df.shape[0]} rows × {st.session_state.processed_df.shape[1]} columns")
    logger.debug("Displaying current dataset info in sidebar")
    
    missing_values = st.session_state.processed_df.isnull().sum().sum()
    if missing_values > 0:
        st.sidebar.warning(f"⚠️ {missing_values} missing values")
        logger.debug(f"Missing values detected: {missing_values}")
    else:
        st.sidebar.success("✅ No missing values")
        logger.debug("No missing values detected")

    # Show data types summary
    st.sidebar.markdown("---")
    st.sidebar.write("**Data Types:**")
    numeric_count = len(st.session_state.processed_df.select_dtypes(include=['number']).columns)
    categorical_count = len(st.session_state.processed_df.select_dtypes(include=['object']).columns)
    other_count = len(st.session_state.processed_df.columns) - numeric_count - categorical_count
    
    st.sidebar.write(f"• Numerical: {numeric_count}")
    st.sidebar.write(f"• Categorical: {categorical_count}")
    if other_count > 0:
        st.sidebar.write(f"• Other: {other_count}")
    
    logger.debug(f"Data types summary: Numerical={numeric_count}, Categorical={categorical_count}, Other={other_count}")

# Add sample datasets for testing
with st.sidebar.expander("💾 Sample Commands"):
    st.write("Try these sample commands:")
    st.code("""
# Basic operations
Clean the data
Show first 10 rows
Describe the data
Dataset info

# Visualizations
Create histogram for age column
Scatter plot of two numerical columns
Bar plot of a categorical column

# Data manipulation
Handle missing values
Filter where age > 30
What columns are available?

# Combined commands
Clean the data and show first 5 rows
Show first 10 rows and create histogram
    """)
    logger.debug("Sample commands displayed")

# Add help section
with st.sidebar.expander("❓ Help"):
    st.write("""
    **How to use:**
    1. Upload a CSV file
    2. Ask questions about your data
    3. The AI assistant will analyze and visualize
    
    **Supported operations:**
    - Data cleaning and preprocessing
    - Statistical analysis
    - Data visualization
    - Filtering and sorting
    - Missing value handling
    
    **Tips:**
    - Be specific with column names
    - Use natural language commands
    - Check the current dataset info in sidebar
    - Use 'Help' command to see available options
    - Combine commands with 'and' or 'then'
    """)
    logger.debug("Help section displayed")

# Add quick action buttons
if st.session_state.processed_df is not None:
    st.sidebar.markdown("---")
    st.sidebar.write("**⚡ Quick Actions:**")
    logger.debug("Quick actions section displayed")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📊 Show First 5", use_container_width=True):
            logger.info("Quick action: Show first 5 rows")
            st.session_state.messages.append({
                "role": "user", 
                "content": "Show first 5 rows", 
                "type": "text"
            })
            response = agent.process_query("Show first 5 rows", st.session_state.processed_df)
            if response["type"] == "dataframe":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "type": "dataframe",
                    "title": response.get("title", "First 5 rows")
                })
            st.rerun()
    
    with col2:
        if st.button("🧹 Clean Data", use_container_width=True):
            logger.info("Quick action: Clean data")
            st.session_state.messages.append({
                "role": "user", 
                "content": "Clean the data", 
                "type": "text"
            })
            response = agent.process_query("Clean the data", st.session_state.processed_df)
            if "df" in response:
                st.session_state.processed_df = response["df"]
                agent.set_current_dataframe(st.session_state.processed_df)
                logger.info("Data cleaned successfully")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["content"],
                "type": "text"
            })
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, LangChain, and Groq API")
logger.debug("Footer displayed")

# Main area tips
if st.session_state.df is None:
    st.info("👆 Please upload a CSV file from the sidebar to get started!")
    logger.debug("Displaying upload prompt in main area")
else:
    st.success(f"✅ Ready to analyze your data! {st.session_state.processed_df.shape[0]} rows × {st.session_state.processed_df.shape[1]} columns loaded.")
    logger.debug("Displaying data ready message")
    
    # Show sample data preview
    with st.expander("📋 Data Preview (first 5 rows)"):
        st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)
        logger.debug("Displaying data preview")

logger.info("Application rendering complete")