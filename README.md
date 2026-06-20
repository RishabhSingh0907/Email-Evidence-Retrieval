# Agentic RAG - End-to-End Retrieval-Augmented Generation Pipeline

## Project Structure

```
Agentic_RAG/
├── src/                          # Source code
│   ├── data_ingestion/           # Stage 1: Extract from raw sources
│   │   └── email_parser_pipeline.py
│   ├── data_processing/          # Stage 2: LLM feature extraction
│   │   └── llm_extractor.py
│   ├── data_loading/             # Stage 3: Load into databases
│   │   └── neo4j_ingestor.py
│   ├── agents/                   # Stage 4: Agent orchestration
│   │   ├── llm_agent_backend.py
│   │   └── rag_agent_backend.py
│   ├── llm/                      # LLM interactions
│   │   └── LLM_canonical_wrapper.py
│   ├── graph/                    # Neo4j graph operations
│   ├── vector_store/             # Vector embeddings & search
│   ├── chat/                     # Chat history management
│   │   └── chat_history_manager.py
│   ├── utils/                    # Shared utilities
│   └── config/                   # Configuration & constants
│       ├── canonical_feature.py
│       └── sample_prompts.txt
├── data/
│   ├── raw/                      # Raw input data
│   │   ├── case_documents/       # Original case documents
│   │   └── new_data/             # Additional raw data
│   ├── processed/                # Intermediate processed data
│   └── vectors/                  # Vector embeddings
├── notebooks/                    # Jupyter notebooks for development
│   └── main.ipynb
├── tests/                        # Unit & integration tests
├── logs/                         # Application logs
├── docs/                         # Documentation
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (keys, credentials)
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

## Testing

```bash
pytest tests/
pytest --cov=src tests/  # With coverage
```

## Documentation

See `docs/` directory for detailed documentation on:
- Architecture overview
- Module specifications
- API reference
- Configuration guide

## TODO / Next Steps

1. Complete vector store implementation
2. Add advanced graph query capabilities
3. Implement semantic caching
4. Add multi-agent coordination
5. Performance optimization

## License

[Add your license information]

## Contact

[Add contact information]
