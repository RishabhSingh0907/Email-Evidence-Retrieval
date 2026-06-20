"""
Main Pipeline Orchestrator
Coordinates all 4 stages of the Agentic RAG pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_load.email_parser_pipeline import process_all_pdfs
# from data_processing.llm_extractor import process_extracted_data
# from data_loading.neo4j_ingestor import ingest_to_neo4j
# from agents.rag_agent_backend import RAGAgent

def run_full_pipeline(pdf_folder: str = "data/raw/case_documents"):
    """
    Run the complete 4-stage RAG pipeline.
    
    Pipeline Flow:
    1. DATA INGESTION: Extract and parse documents
    2. DATA PROCESSING: LLM-based feature extraction
    3. DATA LOADING: Ingest into Neo4j
    4. AGENT: Initialize RAG agent for queries
    """
    
    print("=" * 80)
    print("🚀 AGENTIC RAG PIPELINE - FULL RUN")
    print("=" * 80)
    
    # Stage 1: Data Ingestion
    print("\n📥 STAGE 1: Data Ingestion")
    print("-" * 40)
    try:
        # Adjust path if needed
        if pdf_folder == "data/raw/case_documents":
            full_path = Path(__file__).parent / pdf_folder
        else:
            full_path = pdf_folder
            
        structured_threads = process_all_pdfs(str(full_path))
        print(f"✅ Successfully ingested {len(structured_threads)} documents")
    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        return
    
    # Stage 2: Data Processing
    print("\n🔄 STAGE 2: Data Processing (LLM Extraction)")
    print("-" * 40)
    print("⏳ Not yet implemented. Add llm_extractor integration.")
    # TODO: Uncomment when ready
    # try:
    #     processed_data = process_extracted_data(structured_threads)
    #     print(f"✅ Successfully processed {len(processed_data)} records")
    # except Exception as e:
    #     print(f"❌ Stage 2 failed: {e}")
    #     return
    
    # Stage 3: Data Loading
    print("\n💾 STAGE 3: Data Loading (Neo4j)")
    print("-" * 40)
    print("⏳ Not yet implemented. Add neo4j ingestor integration.")
    # TODO: Uncomment when ready
    # try:
    #     ingest_to_neo4j(processed_data)
    #     print(f"✅ Successfully loaded data into Neo4j")
    # except Exception as e:
    #     print(f"❌ Stage 3 failed: {e}")
    #     return
    
    # Stage 4: Agent Setup
    print("\n🤖 STAGE 4: Agent Initialization")
    print("-" * 40)
    print("⏳ Not yet implemented. Add RAG agent initialization.")
    # TODO: Uncomment when ready
    # try:
    #     agent = RAGAgent()
    #     print(f"✅ RAG Agent initialized and ready for queries")
    # except Exception as e:
    #     print(f"❌ Stage 4 failed: {e}")
    #     return
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the full pipeline
    run_full_pipeline()
