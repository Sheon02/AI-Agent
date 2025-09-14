# Data Analysis AI Agent 🤖
A powerful AI-powered data analysis assistant that provides comprehensive data cleaning, visualization, statistical analysis, and machine learning capabilities through natural language commands.

## 🌟 Features
### **Data Management Tools**
- Clean Data: Handle missing values, remove duplicates, fix data types

- Show Data: Display first/last n rows of dataset

- Describe Data: Statistical description of the dataset

- Get Info: Dataset information including columns and data types

- Filter Data: Filter data based on conditions

- Sample Data: Take random samples from data

- Export Data: Export to CSV, JSON, Excel formats

### **Data Cleaning & Preprocessing**
- Handle Missing Values: Multiple strategies (drop, mean, median)

- Encode Categorical: Convert categorical variables to numerical

- Normalize Data: Scale numerical columns

- Standardize Data: Z-score normalization

- Log Transform: Apply logarithmic transformation

- Outlier Detection: Identify outliers using IQR method

### **Visualization Tools**
- Histogram: Distribution plots for numerical data

- Scatter Plot: Relationships between two numerical variables

- Bar Plot: Categorical data visualization

- Box Plot: Statistical distribution visualization

- Heatmap: Correlation matrix visualization

- Pairplot: Multiple variable relationships

- Violin Plot: Distribution and density visualization

- Pie Chart: Proportional categorical data

- Dashboard: Comprehensive multi-visualization dashboard

### **Statistical Analysis**
- Calculate Correlations: Correlation matrix with highlighting

- Feature Selection: Identify important features for prediction

- Train-Test Split: Data splitting for machine learning

- Cross Validation: Model performance evaluation

### **Machine Learning**
- Select Model: Recommend models based on problem type

- Train Model: Train machine learning models

- Make Predictions: Generate predictions on new data

- Cross Validate: Evaluate model performance

### **Advanced Features**
- Generate Report: Comprehensive analysis reports

- Export Report: Export reports to multiple formats (PDF, HTML, Word, Markdown)

- Search Knowledge: RAG-powered knowledge search for data science concepts

- Generate Code: Python code generation for data analysis tasks

## 🚀 How It Works
### **Architecture**
The AI agent uses a combination of:<br>

- Natural Language Processing: Understands user queries in plain English

- Tool Calling: Dynamically selects appropriate data analysis functions

- RAG (Retrieval Augmented Generation): Web search integration for knowledge queries

- Code Generation: Creates executable Python code for various tasks

### **Workflow**
- User Input: Natural language query (e.g., "create histogram for age column")

- Query Understanding: NLP processing to identify intent and parameters

- Tool Selection: Choose appropriate analysis function

- Execution: Perform the requested data operation

- Response Generation: Return results with visualizations/explanations


## 🏃‍♂️ How to Run
Method 1: Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open your browser to http://localhost:8501<br>

Method 2: Command Line Interface
```bash
python main.py
```

## 📊 Usage Examples
### **Basic Data Operations**
```plaintext
"show first 10 rows"
"clean the dataset"
"describe the data"
"filter where age > 30"
```
### **Visualization Commands**
```plaintext
"create histogram for salary"
"scatter plot of height vs weight"
"bar plot of department distribution"
"create correlation heatmap"
"make a dashboard"
```
### **Advanced Analysis**
```plaintext
"handle missing values with median"
"encode categorical columns"
"detect outliers in the data"
"select top 5 features for prediction"
"train model to predict sales"
```
### **Knowledge Search**
```plaintext
"what are data cleaning steps?"
"how to handle outliers?"
"explain feature engineering"
"what is random forest?"
```
### **Code Generation**
```plaintext
"generate random forest code"
"give me data cleaning code"
"python code for data visualization"
```
## 🛠️ Configuration
### **Environment Variables**
Create a .env file for API keys (optional):<br>

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
SERPAPI_KEY=your_serpapi_key
```
### **Data Format Support**
- CSV Files: Comma-separated values<br>

- Excel Files: .xlsx, .xls formats<br>

- JSON Files: JavaScript Object Notation<br>

- SQL Databases: Through pandas read_sql<br>

- API Data: JSON responses<br>

## 📁 Project Structure
```text
data-analysis-agent/
├── app.py                 # Streamlit web application
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── chatbot/
│   ├── __init__.py
│   ├── agent.py         # Main agent class
│   └── tools.py         # All data analysis tools
├── data/                # Sample datasets
├── outputs/             # Generated reports and exports
└── tests/              # Test files
```
## 🔧 Customization
### **Adding New Tools**
- Add function in tools.py

- Register tool in agent initialization

- Update tool description

### **Modifying Visualizations**
Edit corresponding functions in tools.py to customize:<br>

- Color schemes

- Chart styles

- Layout preferences

- Export formats

## 🚨 Troubleshooting
### **Common Issues**
- Missing Dependencies

```bash
pip install missing-package-name
```
- Memory Issues with Large Datasets

Use sampling: "sample 20% of data"

Enable chunk processing

Increase system memory

- Visualization Not Displaying

Check matplotlib backend

Ensure plotly is installed

- Web Search Not Working

Check internet connection

Verify API keys if using premium services

## Performance Tips
Use sampling for large datasets (>100k rows)

Enable caching for repeated operations

Use appropriate data types to reduce memory usage

## 📈 Output Examples
### **Generated Reports**
- Statistical summaries

- Visualization galleries

- Data quality assessments

- Machine learning performance metrics

- Exportable formats: PDF, HTML, Word, Markdown

### **Visualizations**
- Interactive charts (zoom, pan, hover details)

- High-resolution exports

- Customizable styling

- Multi-plot dashboards

## 🤝 Contributing
- Fork the repository

- Create feature branch (git checkout -b feature/amazing-feature)

- Commit changes (git commit -m 'Add amazing feature')

- Push to branch (git push origin feature/amazing-feature)

- Open a Pull Request


## 🙏 Acknowledgments
- Built with Python data science ecosystem

- Uses Streamlit for web interface

- Integrates multiple AI and search APIs

- Inspired by modern data analysis workflows



Happy Data Analyzing! 📊✨
