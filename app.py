import streamlit as st
import json
import time
import os
import PyPDF2
import requests
import threading
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="LLM Comparison Tool",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("LLM Comparison Tool")
st.markdown("Compare responses from Gemini Flash API and the local NVIDIA GPU server")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys and URLs
    gemini_api_key = st.text_input("Gemini API Key", value="<<update the value here>>", type="password")
    llm_server_url = st.text_input("Local LLM Server URL", value="http://216.48.177.244:8001")
    
    # Output configuration
    save_results = st.checkbox("Save results to file", value=True)
    output_file = st.text_input("Output filename", value="llm_comparison_results.json")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to get response from Gemini
def get_response_from_gemini(document_text, question, api_key):
    genai.configure(api_key=api_key)
    
    prompt = f"""
You are an AI assistant analyzing a document. Please answer the following question based ONLY on the document text provided below.

Document text:
{document_text}

Question: {question}

Please provide a detailed and accurate answer based only on the information present in the document. 
If the answer cannot be found in the document, state so clearly.
"""
    
    try:
        # Record start time
        start_time = time.time()
        
        # Safety settings for Gemini
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        
        # Generation parameters
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        return response.text, response_time
            
    except Exception as e:
        return f"Error: {str(e)}", 0

# Function to get response from local LLM
def get_response_from_local_llm(document_text, question, server_url):
    generate_url = f"{server_url}/generate"
    
    prompt = f"""
You are an AI assistant analyzing a document. Please answer the following question based ONLY on the document text provided below.

Document text:
{document_text}

Question: {question}

Please provide a detailed and accurate answer based only on the information present in the document. 
If the answer cannot be found in the document, state so clearly.
"""
    
    payload = {
        "prompt": prompt,
        "max_tokens": 4000,
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    try:
        # Record start time
        start_time = time.time()
        
        # Make API call
        response = requests.post(generate_url, json=payload, timeout=60)
        response.raise_for_status()
        generation_data = response.json()
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        if "response" in generation_data:
            return generation_data["response"], response_time
        else:
            return "Error: No response found in LLM generation data", response_time
            
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}", 0

# Function to compare responses
def compare_responses(gemini_response, llm_response, gemini_time, llm_time):
    # Get word count
    gemini_words = len(gemini_response.split())
    llm_words = len(llm_response.split())
    
    # Get paragraphs
    gemini_paragraphs = [p.strip() for p in gemini_response.split('\n\n') if p.strip()]
    llm_paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip()]
    
    # Calculate Jaccard similarity
    gemini_set = set(gemini_response.lower().split())
    llm_set = set(llm_response.lower().split())
    
    intersection = len(gemini_set.intersection(llm_set))
    union = len(gemini_set.union(llm_set))
    
    jaccard_similarity = intersection / union if union > 0 else 0
    
    return {
        "length_comparison": f"Gemini response: {gemini_words} words\nLocal LLM response: {llm_words} words",
        "paragraph_comparison": f"Gemini paragraphs: {len(gemini_paragraphs)}\nLocal LLM paragraphs: {len(llm_paragraphs)}",
        "time_comparison": f"Gemini response time: {gemini_time:.2f} seconds\nLocal LLM response time: {llm_time:.2f} seconds",
        "similarity": f"Vocabulary similarity (Jaccard index): {jaccard_similarity:.4f}",
        "gemini_words": gemini_words,
        "llm_words": llm_words,
        "gemini_paragraphs": len(gemini_paragraphs),
        "llm_paragraphs": len(llm_paragraphs),
        "gemini_time": gemini_time,
        "llm_time": llm_time,
        "jaccard_similarity": jaccard_similarity
    }

# Function to get responses concurrently
def get_responses_concurrently(document_text, question, gemini_api_key, llm_server_url):
    gemini_result = [None, 0]
    llm_result = [None, 0]
    
    def gemini_thread():
        response, time_taken = get_response_from_gemini(document_text, question, gemini_api_key)
        gemini_result[0] = response
        gemini_result[1] = time_taken
    
    def llm_thread():
        response, time_taken = get_response_from_local_llm(document_text, question, llm_server_url)
        llm_result[0] = response
        llm_result[1] = time_taken
    
    # Create and start threads
    t1 = threading.Thread(target=gemini_thread)
    t2 = threading.Thread(target=llm_thread)
    
    t1.start()
    t2.start()
    
    # Wait for threads to complete
    t1.join()
    t2.join()
    
    return gemini_result[0], gemini_result[1], llm_result[0], llm_result[1]

# Main app logic
tab1, tab2 = st.tabs(["Single Document", "Batch Processing"])

with tab1:
    st.header("Single Document Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Text area for direct input
    use_text_input = st.checkbox("Or enter document text directly")
    document_text = ""
    
    if use_text_input:
        document_text = st.text_area("Enter document text", height=200)
    
    # Question input
    question = st.text_input("Enter your question about the document")
    
    # Process button
    if st.button("Compare Responses", key="compare_single"):
        if (uploaded_file or document_text) and question:
            with st.spinner("Processing..."):
                # Get document text
                if uploaded_file and not use_text_input:
                    document_text = extract_text_from_pdf(uploaded_file)
                    st.info(f"Extracted {len(document_text)} characters from PDF")
                
                if not document_text:
                    st.error("No document text available. Please upload a PDF or enter text directly.")
                else:
                    # Get responses concurrently
                    gemini_response, gemini_time, llm_response, llm_time = get_responses_concurrently(
                        document_text, question, gemini_api_key, llm_server_url
                    )
                    
                    # Compare responses
                    comparison = compare_responses(gemini_response, llm_response, gemini_time, llm_time)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Gemini Flash API")
                        st.info(f"Response time: {gemini_time:.2f} seconds")
                        st.write(gemini_response)
                    
                    with col2:
                        st.subheader("NVIDIA GPU Server (Local LLM)")
                        st.info(f"Response time: {llm_time:.2f} seconds")
                        st.write(llm_response)
                    
                    # Display comparison
                    st.subheader("Comparison")
                    st.write(comparison["length_comparison"])
                    st.write(comparison["paragraph_comparison"])
                    st.write(comparison["time_comparison"])
                    st.write(comparison["similarity"])
                    
                    # Create speedometer chart for time comparison
                    st.subheader("Response Time Comparison")
                    max_time = max(gemini_time, llm_time)
                    col1, col2 = st.columns(2)
                    col1.metric("Gemini Flash API", f"{gemini_time:.2f}s")
                    col2.metric("NVIDIA GPU Server", f"{llm_time:.2f}s", 
                               f"{(llm_time - gemini_time):.2f}s" if llm_time > gemini_time else f"{(gemini_time - llm_time):.2f}s")
                    
                    # Save results if requested
                    if save_results:
                        results = {
                            "file_info": {
                                "filename": uploaded_file.name if uploaded_file else "Text input",
                                "size_kb": uploaded_file.size / 1024 if uploaded_file else len(document_text) / 1024,
                                "text_length": len(document_text)
                            },
                            "question": question,
                            "gemini_response": gemini_response,
                            "gemini_time": gemini_time,
                            "local_llm_response": llm_response,
                            "local_llm_time": llm_time,
                            "comparison": comparison
                        }
                        
                        # Write detailed results
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2)
                        
                        # Write simplified comparison
                        simplified_results = {
                            "nvidia_gpu_server": {
                                "prompt": question,
                                "response": llm_response,
                                "time_taken (s)": round(llm_time, 1)
                            },
                            "gemini_flash_api": {
                                "prompt": question,
                                "response": gemini_response,
                                "time_taken (s)": round(gemini_time, 1)
                            }
                        }
                        
                        simplified_output = os.path.splitext(output_file)[0] + "_comparison.json"
                        with open(simplified_output, 'w', encoding='utf-8') as f:
                            json.dump(simplified_results, f, indent=2)
                        
                        st.success(f"Results saved to {output_file} and {simplified_output}")
        else:
            st.error("Please provide a document and a question.")

with tab2:
    st.header("Batch Processing")
    st.write("Upload multiple PDF files for batch processing.")
    
    # File upload for batch processing
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    
    # Question input for batch processing
    batch_question = st.text_input("Enter your question for all documents", key="batch_question")
    
    # Batch processing output directory
    output_dir = st.text_input("Output directory", value="batch_results")
    
    # Process button for batch
    if st.button("Process Batch", key="process_batch"):
        if uploaded_files and batch_question:
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each file
                all_results = {}
                all_simplified_results = {}
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    # Extract text from PDF
                    document_text = extract_text_from_pdf(uploaded_file)
                    
                    if document_text:
                        # Get responses concurrently
                        gemini_response, gemini_time, llm_response, llm_time = get_responses_concurrently(
                            document_text, batch_question, gemini_api_key, llm_server_url
                        )
                        
                        # Compare responses
                        comparison = compare_responses(gemini_response, llm_response, gemini_time, llm_time)
                        
                        # Prepare results
                        results = {
                            "file_info": {
                                "filename": uploaded_file.name,
                                "size_kb": uploaded_file.size / 1024,
                                "text_length": len(document_text)
                            },
                            "question": batch_question,
                            "gemini_response": gemini_response,
                            "gemini_time": gemini_time,
                            "local_llm_response": llm_response,
                            "local_llm_time": llm_time,
                            "comparison": comparison
                        }
                        
                        # Save individual results
                        output_file = os.path.join(output_dir, f"{os.path.splitext(uploaded_file.name)[0]}_results.json")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2)
                        
                        # Add to summaries
                        all_results[uploaded_file.name] = results
                        all_simplified_results[uploaded_file.name] = {
                            "nvidia_gpu_server": {
                                "prompt": batch_question,
                                "response": llm_response,
                                "time_taken (s)": round(llm_time, 1)
                            },
                            "gemini_flash_api": {
                                "prompt": batch_question,
                                "response": gemini_response,
                                "time_taken (s)": round(gemini_time, 1)
                            }
                        }
                    else:
                        all_results[uploaded_file.name] = {"error": "Failed to extract text from PDF"}
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Save summary results
                summary_file = os.path.join(output_dir, "batch_summary.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                
                # Save simplified comparison
                simplified_summary_file = os.path.join(output_dir, "batch_comparison.json")
                with open(simplified_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(all_simplified_results, f, indent=2)
                
                # Display completion message
                status_text.text("Batch processing complete!")
                st.success(f"Results saved to {output_dir}")
                
                # Calculate and display batch statistics
                successful = sum(1 for r in all_results.values() if "error" not in r)
                st.write(f"Processed {len(uploaded_files)} files, {successful} successful, {len(uploaded_files) - successful} failed")
                
                if successful > 0:
                    avg_gemini_time = sum(r.get('gemini_time', 0) for r in all_results.values() if "error" not in r) / successful
                    avg_llm_time = sum(r.get('local_llm_time', 0) for r in all_results.values() if "error" not in r) / successful
                    avg_similarity = sum(r.get('comparison', {}).get('jaccard_similarity', 0) for r in all_results.values() if "error" not in r) / successful
                    
                    st.write(f"Average response times:")
                    col1, col2 = st.columns(2)
                    col1.metric("Gemini Flash API", f"{avg_gemini_time:.2f}s")
                    col2.metric("NVIDIA GPU Server", f"{avg_llm_time:.2f}s")
                    st.metric("Average vocabulary similarity", f"{avg_similarity:.4f}")
        else:
            st.error("Please upload at least one PDF file and provide a question.")

# Add a footer
st.markdown("---")
st.markdown("LLM Comparison Tool - Comparing Gemini Flash API and local NVIDIA GPU server responses")
