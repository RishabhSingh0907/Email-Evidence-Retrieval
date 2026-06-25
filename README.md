# Agentic RAG - End-to-End Retrieval-Augmented Generation Pipeline

## Project Structure

```
Agentic_RAG/
├── src/                          # Source code
│   ├── data_ingest/           # Stage 1: Extract from raw sources
│   │   └── neo4j_ingestor.py
│   ├── data_processing/          # Stage 2: LLM feature extraction
│   │   └── llm_extractor.py
│   │   └── embedding_model.py
│   ├── data_load/             # Stage 3: Load into databases
│   │   └── email_parser_pipeline.py
│   ├── agents/                   # Stage 4: Agent orchestration
│   │   ├── agent.py
│   │   └── graph.py
|   |   └── llm.py
│   ├── graph/                    # Neo4j graph operations
│   │   └── graph_schema_visualization.py
│   ├── chat/                     # Chat history management
│   │   └── app.py
│   │   └── chat_history_manager.py
│   │   └── response_formatter.py
│   ├── tools/                    # Shared utilities
│   │   └── cypher_tools.py
│   │   └── semantic_tool.py
│   │   └── text2cypher_tool.py
├── data/
│   ├── raw/                      # Raw input data
│   │   ├── case_documents/       # Original case documents
│   ├── processed/                # Intermediate processed data
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
├── pipeline.py                          # pipeline for ingestion and loading data into database
└── README.md                     # Project documentation
```

## Pipeline Stages

### Stage 1: Data Ingestion
**Module:** `src/data_ingestion/email_parser_pipeline.py`
- Extracts text from PDF documents
- Parses email/thread structure
- Normalizes whitespace and formatting
- Output: Structured message blocks with IDs and relationships

### Stage 2: Data Processing
**Module:** `src/data_processing/llm_extractor.py`
- Uses LLM to extract features and entities
- Identifies key information (people, dates, events, amounts)
- Classifies message intent and sentiment
- Output: Enriched structured data with extracted features

### Stage 3: Data Loading
**Module:** `src/data_loading/neo4j_ingestor.py`
- Ingests processed data into Neo4j graph database
- Creates entity and relationship nodes
- Builds knowledge graph from extracted features
- Output: Populated Neo4j database

### Stage 4: Agent Orchestration
**Modules:** `src/agents/llm_agent_backend.py`, `src/agents/rag_agent_backend.py`
- Orchestrates the full RAG pipeline
- Implements agent-based reasoning
- Handles query processing and response generation
- Output: User-facing conversational interface

## Setup & Installation

1. **Clone the repository**
   ```bash
   cd Agentic_RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or: source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`
   - Add API keys (OpenAI, Neo4j credentials, etc.)
   - Update paths if needed

5. **Run the pipeline**
   ```bash
   # Option 1: Run full pipeline
   python app.py
   
   # Option 2: Run individual stages
   python src/data_ingestion/email_parser_pipeline.py
   python src/data_processing/llm_extractor.py
   python src/data_loading/neo4j_ingestor.py
   python src/agents/rag_agent_backend.py
   ```

## Key Features

- 📧 **Email Parser**: Extracts and structures email threads from PDFs
- 🤖 **LLM Integration**: Uses language models for intelligent feature extraction
- 📊 **Knowledge Graph**: Builds Neo4j graph for relationship querying
- 🔍 **Semantic Search**: Vector embeddings for efficient retrieval
- 💬 **Conversational Agent**: Chat interface with context awareness
- 📈 **Logging**: Comprehensive logging for debugging and monitoring

## License

[Add your license information]

## Contact

[Add contact information]
