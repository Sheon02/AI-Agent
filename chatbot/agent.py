from langchain_groq import ChatGroq
from langchain.agents import Tool
from .tools import DataAnalysisTools
import os
import pandas as pd
import re

class DataAnalystAgent:
    def __init__(self):
        self.tools_instance = DataAnalysisTools()
    
    def process_query(self, query: str, df) -> dict:
        """Process user query and return appropriate response"""
        print(f"🔍 DEBUG: process_query called with query: '{query}'")
        print(f"🔍 DEBUG: DataFrame shape: {df.shape if df is not None else 'None'}")
        
        # Set the current dataframe in the tools instance
        self.tools_instance.current_df = df
        
        query_lower = query.lower()
        print(f"🔍 DEBUG: Query lowercased: '{query_lower}'")
        
        try:
            # Check for dashboard requests FIRST (before multiple commands)
            dashboard_keywords = [
                'dashboard', 'overview', 'missing', 'correlation', 'distribution', 
                'quality', 'statistic', 'recommend', 'summary', 'analysis'
            ]
            
            # Check if this is a dashboard-related request
            is_dashboard_request = any(keyword in query_lower for keyword in dashboard_keywords)
            
            # If it's a dashboard request, handle it as a single command
            if is_dashboard_request:
                print("🔍 DEBUG: Dashboard request detected, calling create_dashboard")
                return self.tools_instance.create_dashboard(query)
        
            # Check for multiple commands separated by "and" or "then"
            if " and " in query_lower or " then " in query_lower:
                print("🔍 DEBUG: Multiple commands detected, processing with _process_multiple_commands")
                return self._process_multiple_commands(query, df)
            
            # Single command processing
            # SPECIFIC COMMANDS FIRST (more specific to less specific)
            
            # REPORT GENERATION TOOLS
            if "report" in query_lower and ("generate" in query_lower or "create" in query_lower):
                print("🔍 DEBUG: Calling generate_report")
                return self.tools_instance.generate_report(query)
            
            elif "export" in query_lower and "report" in query_lower:
                print("🔍 DEBUG: Calling export_report")
                return self.tools_instance.export_report(query)
            
            # DATA TRANSFORMATION TOOLS
            elif "log" in query_lower and "transform" in query_lower:
                print("🔍 DEBUG: Calling log_transform")
                return self.tools_instance.log_transform(query)
            
            elif "standardize" in query_lower or "z-score" in query_lower:
                print("🔍 DEBUG: Calling standardize_data")
                return self.tools_instance.standardize_data(query)
            
            # VISUALIZATION TOOLS
            elif "pairplot" in query_lower or "pair plot" in query_lower:
                print("🔍 DEBUG: Calling create_pairplot")
                return self.tools_instance.create_pairplot(query)
            
            elif "violin" in query_lower and "plot" in query_lower:
                print("🔍 DEBUG: Calling create_violinplot")
                return self.tools_instance.create_violinplot(query)
            
            # STATISTICAL ANALYSIS TOOLS
            elif "correlation" in query_lower and ("calculate" in query_lower or "show" in query_lower):
                print("🔍 DEBUG: Calling calculate_correlations")
                return self.tools_instance.calculate_correlations(query)
            
            elif "outlier" in query_lower and ("detect" in query_lower or "find" in query_lower):
                print("🔍 DEBUG: Calling outlier_detection")
                return self.tools_instance.outlier_detection(query)
            
            # DATA SAMPLING TOOLS
            elif "sample" in query_lower and ("data" in query_lower or "rows" in query_lower):
                print("🔍 DEBUG: Calling sample_data")
                return self.tools_instance.sample_data(query)
            
            elif ("train" in query_lower and "test" in query_lower) or "split" in query_lower:
                print("🔍 DEBUG: Calling train_test_split")
                return self.tools_instance.train_test_split(query)
            
            # DATA EXPORT TOOLS
            elif "export" in query_lower and ("csv" in query_lower or "json" in query_lower or "excel" in query_lower):
                print("🔍 DEBUG: Calling export_data")
                return self.tools_instance.export_data(query)
            
            elif any(keyword in query_lower for keyword in ['how to', 'what is', 'steps for','steps in', 'guide for', 'explain']):
                print("🔍 DEBUG: Calling search_knowledge")
                return self.tools_instance.search_knowledge(query)

            elif any(keyword in query_lower for keyword in ['code', 'generate code', 'python code', 'script']):
                print("🔍 DEBUG: Calling generate_code")
                return self.tools_instance.generate_code(query)
            
            # Replace the existing dashboard condition with:
            elif any(keyword in query_lower for keyword in [
                'dashboard', 'overview', 'missing', 'correlation', 'distribution', 
                'quality', 'statistic', 'recommend', 'summary'
            ]):
                print("🔍 DEBUG: Calling create_dashboard")
                return self.tools_instance.create_dashboard(query)
            
            # EXISTING TOOLS (keep the original order for existing tools)
            elif "create" in query_lower and "new" in query_lower and "column" in query_lower:
                print("🔍 DEBUG: Calling create_new_column")
                return self.tools_instance.create_new_column(query)
            
            elif "select" in query_lower and "feature" in query_lower:
                print("🔍 DEBUG: Calling select_features")
                return self.tools_instance.select_features(query)
            
            elif "encode" in query_lower or "convert" in query_lower or "transform" in query_lower:
                if "categorical" in query_lower or "category" in query_lower or any(col in query_lower for col in self._get_categorical_columns(df)):
                    print("🔍 DEBUG: Calling encode_categorical")
                    return self.tools_instance.encode_categorical(query)
                else:
                    print("🔍 DEBUG: Generic encode request, checking if categorical columns exist")
                    categorical_cols = self._get_categorical_columns(df)
                    if categorical_cols:
                        print("🔍 DEBUG: Found categorical columns, calling encode_categorical")
                        return self.tools_instance.encode_categorical(query)
                    else:
                        return {"type": "text", "content": "No categorical columns found to encode."}
            
            elif "normalize" in query_lower:
                print("🔍 DEBUG: Calling normalize_data")
                return self.tools_instance.normalize_data(query)
            
            elif "clean" in query_lower and "data" in query_lower:
                print("🔍 DEBUG: Calling _process_clean_data")
                return self._process_clean_data(query)
            
            elif "show" in query_lower and ("first" in query_lower or "rows" in query_lower or "last" in query_lower):
                print("🔍 DEBUG: Calling show_data")
                return self.tools_instance.show_data(query)
            
            elif "histogram" in query_lower:
                print("🔍 DEBUG: Calling create_histogram")
                return self.tools_instance.create_histogram(query)
            
            elif "scatter" in query_lower or ("plot" in query_lower and "scatter" not in query_lower):
                print("🔍 DEBUG: Calling create_scatterplot")
                return self.tools_instance.create_scatterplot(query)
            
            elif "bar" in query_lower and "plot" in query_lower:
                print("🔍 DEBUG: Calling create_barplot")
                return self.tools_instance.create_barplot(query)
            
            elif "box" in query_lower and "plot" in query_lower:
                print("🔍 DEBUG: Calling create_boxplot")
                return self.tools_instance.create_boxplot(query)
            
            elif "heatmap" in query_lower or "correlation" in query_lower:
                print("🔍 DEBUG: Calling create_heatmap")
                return self.tools_instance.create_heatmap(query)
            
            elif "describe" in query_lower:
                print("🔍 DEBUG: Calling describe_data")
                return self.tools_instance.describe_data(query)
            
            # Handle info/column queries - MAKE THIS MORE SPECIFIC
            elif ("info" in query_lower and "dataset" in query_lower) or ("show" in query_lower and "info" in query_lower):
                print("🔍 DEBUG: Calling get_info")
                return self.tools_instance.get_info(query)
            
            elif "column" in query_lower and ("available" in query_lower or "list" in query_lower or "name" in query_lower or "what" in query_lower):
                print("🔍 DEBUG: Calling get_column_names")
                return self.tools_instance.get_column_names(query)
            
            elif "missing" in query_lower or "null" in query_lower:
                print("🔍 DEBUG: Calling handle_missing_values")
                return self.tools_instance.handle_missing_values(query)
            
            elif "filter" in query_lower:
                print("🔍 DEBUG: Calling filter_data")
                return self.tools_instance.filter_data(query)
            
            elif "help" in query_lower or "what can you do" in query_lower:
                print("🔍 DEBUG: Calling _get_help_response")
                return self._get_help_response()
            
            elif "pie" in query_lower and "chart" in query_lower:
                print("🔍 DEBUG: Calling create_piechart")
                return self.tools_instance.create_piechart(query)

            elif "model" in query_lower and ("select" in query_lower or "recommend" in query_lower):
                print("🔍 DEBUG: Calling select_model")
                return self.tools_instance.select_model(query)

            elif "train" in query_lower and "model" in query_lower:
                print("🔍 DEBUG: Calling train_model")
                return self.tools_instance.train_model(query)

            elif "predict" in query_lower and ("make" in query_lower or "generate" in query_lower):
                print("🔍 DEBUG: Calling make_predictions")
                return self.tools_instance.make_predictions(query)

            elif "cross" in query_lower and "validate" in query_lower:
                print("🔍 DEBUG: Calling cross_validate_model")
                return self.tools_instance.cross_validate_model(query)
            
            else:
                print("🔍 DEBUG: No matching command found, returning help message")
                return {"type": "text", "content": "I'm not sure how to process that request. Please try a specific data analysis command or ask for help."}
                
        except Exception as e:
            error_msg = f"❌ Error processing request: {str(e)}"
            print(f"🔍 DEBUG: Exception in process_query: {error_msg}")
            return {"type": "text", "content": error_msg}
    
    def _get_categorical_columns(self, df):
        """Get categorical columns from dataframe"""
        if df is None:
            return []
        return df.select_dtypes(include=['object']).columns.tolist()
    
    def _process_multiple_commands(self, query: str, df) -> dict:
        """Process multiple commands in a single query"""
        print(f"🔍 DEBUG: _process_multiple_commands called with: '{query}'")
        # Check if this is actually a dashboard request in disguise
        dashboard_keywords = [
            'dashboard', 'overview', 'missing', 'correlation', 'distribution', 
            'quality', 'statistic', 'recommend', 'summary', 'analysis'
        ]
        
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in dashboard_keywords):
            print("🔍 DEBUG: This is a dashboard request, not multiple commands")
            return self.tools_instance.create_dashboard(query)
        
        # Only proceed with multiple commands if it's not a dashboard request
        # Split the query into individual commands
        if " and " in query.lower():
            commands = [cmd.strip() for cmd in query.split(" and ")]
        elif " then " in query.lower():
            commands = [cmd.strip() for cmd in query.split(" then ")]
        else:
            # Try to split by other conjunctions
            commands = [cmd.strip() for cmd in re.split(r',|;|&', query)]
        
        print(f"🔍 DEBUG: Split commands: {commands}")
        
        results = []
        current_df = df.copy()
        
        for i, command in enumerate(commands):
            if not command.strip():
                continue
                
            print(f"🔍 DEBUG: Processing command {i+1}: '{command}'")
            
            # Set the current dataframe for this command
            self.tools_instance.current_df = current_df
            
            # Process each command
            command_lower = command.lower()
            
            try:
                if "report" in command_lower and ("generate" in command_lower or "create" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling generate_report")
                    result = self.tools_instance.generate_report(command)
                    results.append(result)
            
                elif "log" in command_lower and "transform" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling log_transform")
                    result = self.tools_instance.log_transform(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
            
                elif "standardize" in command_lower or "z-score" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling standardize_data")
                    result = self.tools_instance.standardize_data(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
            
                elif "pairplot" in command_lower or "pair plot" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_pairplot")
                    result = self.tools_instance.create_pairplot(command)
                    results.append(result)
            
                elif "violin" in command_lower and "plot" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_violinplot")
                    result = self.tools_instance.create_violinplot(command)
                    results.append(result)
            
                elif "correlation" in command_lower and ("calculate" in command_lower or "show" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling calculate_correlations")
                    result = self.tools_instance.calculate_correlations(command)
                    results.append(result)
            
                elif "outlier" in command_lower and ("detect" in command_lower or "find" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling outlier_detection")
                    result = self.tools_instance.outlier_detection(command)
                    results.append(result)
            
                elif "sample" in command_lower and ("data" in command_lower or "rows" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling sample_data")
                    result = self.tools_instance.sample_data(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
            
                elif ("train" in command_lower and "test" in command_lower) or "split" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling train_test_split")
                    result = self.tools_instance.train_test_split(command)
                    if "train_df" in result and "test_df" in result:
                        # For simplicity, keep the train_df as current
                        current_df = result["train_df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
            
                elif "export" in command_lower and ("csv" in command_lower or "json" in command_lower or "excel" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling export_data")
                    result = self.tools_instance.export_data(command)
                    results.append(result)
            
                if "clean" in command_lower and "data" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling clean_data")
                    result = self.tools_instance.clean_data(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
                
                elif "show" in command_lower and ("first" in command_lower or "rows" in command_lower or "last" in command_lower):
                    print(f"🔍 DEBUG: Command {i+1}: Calling show_data")
                    result = self.tools_instance.show_data(command)
                    results.append(result)
                
                elif "histogram" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_histogram")
                    result = self.tools_instance.create_histogram(command)
                    results.append(result)
                
                elif "scatter" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_scatterplot")
                    result = self.tools_instance.create_scatterplot(command)
                    results.append(result)
                
                elif "bar" in command_lower and "plot" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_barplot")
                    result = self.tools_instance.create_barplot(command)
                    results.append(result)
                
                elif "box" in command_lower and "plot" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_boxplot")
                    result = self.tools_instance.create_boxplot(command)
                    results.append(result)
                
                elif "heatmap" in command_lower or "correlation" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_heatmap")
                    result = self.tools_instance.create_heatmap(command)
                    results.append(result)
                
                elif "describe" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling describe_data")
                    result = self.tools_instance.describe_data(command)
                    results.append(result)
                
                elif "info" in command_lower or "column" in command_lower:
                    if "column" in command_lower and ("available" in command_lower or "list" in command_lower or "name" in command_lower):
                        print(f"🔍 DEBUG: Command {i+1}: Calling get_column_names")
                        result = self.tools_instance.get_column_names(command)
                    else:
                        print(f"🔍 DEBUG: Command {i+1}: Calling get_info")
                        result = self.tools_instance.get_info(command)
                    results.append(result)
                
                # ENHANCED ENCODING HANDLING FOR MULTIPLE COMMANDS
                elif "encode" in command_lower or "convert" in command_lower or "transform" in command_lower:
                    if "categorical" in command_lower or "category" in command_lower or any(col in command_lower for col in self._get_categorical_columns(current_df)):
                        print(f"🔍 DEBUG: Command {i+1}: Calling encode_categorical")
                        result = self.tools_instance.encode_categorical(command)
                        if "df" in result:
                            current_df = result["df"]
                            print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                        results.append(result)
                    else:
                        result = {"type": "text", "content": f"No categorical columns specified or found for command: {command}"}
                        results.append(result)
                
                elif "normalize" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling normalize_data")
                    result = self.tools_instance.normalize_data(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
                
                elif "create" in command_lower and "new" in command_lower and "column" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling create_new_column")
                    result = self.tools_instance.create_new_column(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
                
                elif "select" in command_lower and "feature" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling select_features")
                    result = self.tools_instance.select_features(command)
                    results.append(result)
                
                elif "missing" in command_lower or "null" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling handle_missing_values")
                    result = self.tools_instance.handle_missing_values(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
                
                elif "filter" in command_lower:
                    print(f"🔍 DEBUG: Command {i+1}: Calling filter_data")
                    result = self.tools_instance.filter_data(command)
                    if "df" in result:
                        current_df = result["df"]
                        print(f"🔍 DEBUG: DataFrame updated to shape: {current_df.shape}")
                    results.append(result)
                
                else:
                    print(f"🔍 DEBUG: Command {i+1}: No matching tool found")
                    results.append({"type": "text", "content": f"Could not process command: {command}"})
            
            except Exception as e:
                error_msg = f"Error processing '{command}': {str(e)}"
                print(f"🔍 DEBUG: Command {i+1} Exception: {error_msg}")
                results.append({"type": "text", "content": error_msg})
        
        print(f"🔍 DEBUG: All commands processed. Final DataFrame shape: {current_df.shape}")
        print(f"🔍 DEBUG: Results types: {[r['type'] for r in results]}")
        
        # Combine results
        return self._combine_multiple_results(results, current_df)
    
    def _handle_combined_histogram_request(self, query: str, df) -> dict:
        """Special handling for show rows + histogram requests"""
        print(f"🔍 DEBUG: _handle_combined_histogram_request called with: '{query}'")
        
        try:
            self.tools_instance.current_df = df
            
            # Extract the number of rows to show
            n = 5
            num_match = re.search(r'(\d+)\s*rows?', query.lower())
            if num_match:
                n = int(num_match.group(1))
            
            print(f"🔍 DEBUG: Showing first {n} rows")
            show_result = self.tools_instance.show_data(f"show first {n} rows")
            print(f"🔍 DEBUG: Show result type: {show_result['type']}")
            
            # Create histogram using the full query to preserve column context
            print(f"🔍 DEBUG: Creating histogram with query: {query}")
            histogram_result = self.tools_instance.create_histogram(query)
            print(f"🔍 DEBUG: Histogram result type: {histogram_result['type']}")
            
            # If histogram failed, provide helpful error
            if histogram_result["type"] == "text":
                print(f"🔍 DEBUG: Histogram failed: {histogram_result['content']}")
                response_content = f"{show_result.get('content', 'Data shown below')}\n\n❌ Histogram creation failed: {histogram_result['content']}"
                
                combined_response = {
                    "type": "text",
                    "content": response_content
                }
                
                if show_result["type"] == "dataframe":
                    combined_response["dataframe_preview"] = show_result["content"]
                
                return combined_response
            
            # Combine successful results
            response_content = f"Showing first {n} rows and histogram:\n\n"
            
            if show_result["type"] == "dataframe" and histogram_result["type"] == "image":
                print("🔍 DEBUG: Both show and histogram successful")
                return {
                    "type": "text",
                    "content": response_content,
                    "dataframe_preview": show_result["content"],
                    "image_content": histogram_result["content"],
                    "title": histogram_result.get("title", "Histogram")
                }
            else:
                print("🔍 DEBUG: Mixed results from show and histogram")
                response_content = ""
                if show_result["type"] == "text":
                    response_content += f"Data: {show_result['content']}\n\n"
                if histogram_result["type"] == "text":
                    response_content += f"Histogram: {histogram_result['content']}"
                
                combined_response = {
                    "type": "text",
                    "content": response_content.strip()
                }
                
                if show_result["type"] == "dataframe":
                    combined_response["dataframe_preview"] = show_result["content"]
                if histogram_result["type"] == "image":
                    combined_response["image_content"] = histogram_result["content"]
                
                return combined_response
                
        except Exception as e:
            error_msg = f"❌ Error processing combined request: {str(e)}"
            print(f"🔍 DEBUG: Exception in combined handler: {error_msg}")
            return {"type": "text", "content": error_msg}
    
    def _process_clean_data(self, query: str) -> dict:
        """Special handling for clean data command"""
        print(f"🔍 DEBUG: _process_clean_data called with: '{query}'")
        
        clean_result = self.tools_instance.clean_data(query)
        
        if "df" in clean_result:
            print(f"🔍 DEBUG: Clean successful, DataFrame shape: {clean_result['df'].shape}")
            
            # If clean was successful, check if we should show data too
            if "show" in query.lower() or "display" in query.lower():
                print("🔍 DEBUG: Clean + show requested")
                # Temporarily set the cleaned dataframe to show results
                original_df = self.tools_instance.current_df
                self.tools_instance.current_df = clean_result["df"]
                
                show_result = self.tools_instance.show_data("show first 5 rows")
                
                # Restore original dataframe
                self.tools_instance.current_df = original_df
                
                if show_result["type"] == "dataframe":
                    return {
                        "type": "text",
                        "content": clean_result["content"],
                        "df": clean_result["df"],
                        "dataframe_preview": show_result["content"]
                    }
        
        return clean_result
    
    def _combine_multiple_results(self, results: list, final_df: pd.DataFrame) -> dict:
        """Combine results from multiple commands"""
        print(f"🔍 DEBUG: _combine_multiple_results called with {len(results)} results")
        
        combined_content = ""
        dataframe_preview = None
        image_content = None
        
        for i, result in enumerate(results):
            print(f"🔍 DEBUG: Result {i+1} type: {result['type']}")
            if result["type"] == "text":
                combined_content += result["content"] + "\n\n"
            elif result["type"] == "dataframe":
                dataframe_preview = result["content"]
            elif result["type"] == "image":
                image_content = result["content"]
        
        response = {"type": "text", "content": combined_content.strip(), "df": final_df}
        
        if dataframe_preview is not None:
            response["dataframe_preview"] = dataframe_preview
        
        if image_content is not None:
            response["image_content"] = image_content
            response["type"] = "image"
        
        print(f"🔍 DEBUG: Combined response type: {response['type']}")
        return response
    
    def _get_help_response(self) -> dict:
        """Get help response"""
        print("🔍 DEBUG: Returning help response")
        return {
            "type": "text",
            "content": """I can help you with various data analysis tasks:

            📊 **Data Operations:**
            - Clean data (remove duplicates, handle missing values)
            - Show data preview (first/last n rows)
            - Describe statistical properties
            - Filter data based on conditions
            - Handle missing values
            - Sample data (random subsets)
            - Train-test split for ML

            📈 **Visualizations:**
            - Create histograms for numerical columns
            - Create scatter plots between two numerical columns
            - Create bar plots for categorical columns
            - Create box plots for numerical data
            - Create correlation heatmaps
            - Create pairplots for multiple variables
            - Create violin plots for distribution comparison

            🔧 **Feature Engineering:**
            - Encode categorical variables (specific columns or all)
            - Normalize numerical data
            - Standardize data (z-score normalization)
            - Apply log transformations
            - Create new columns from expressions
            - Select important features

            📋 **Statistical Analysis:**
            - Calculate correlations and identify strong relationships
            - Detect outliers using IQR method
            - Generate comprehensive data analysis reports

            💾 **Export & Reporting:**
            - Export data to CSV, JSON, Excel formats
            - Generate comprehensive analysis reports
            - Export reports to various formats

            ℹ️ **Information:**
            - Show dataset information and columns
            - Display data types and missing values

            💡 **Multiple Commands:**
            You can combine commands using "and" or "then":
            - "Clean data and show first 5 rows"
            - "Generate report and export to PDF"
            - "Detect outliers then filter them out"
            - "Encode categorical columns then train-test split"

            Try commands like:
            - "Generate comprehensive report"
            - "Create pairplot of numerical columns"
            - "Detect outliers in salary column"
            - "Standardize numerical data"
            - "Sample 20% of the data"
            - "Split data 80/20 for train-test"
            - "Export data to CSV"
            """
            }
    
    def set_current_dataframe(self, df):
        """Set the current dataframe for all tools"""
        print(f"🔍 DEBUG: set_current_dataframe called with shape: {df.shape if df is not None else 'None'}")
        self.tools_instance.current_df = df