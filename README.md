# Retrieval-Augmented Generation System for YouTube Videos and Comments

A comprehensive RAG (Retrieval-Augmented Generation) system that enhances language model capabilities by integrating YouTube video transcripts and comments to provide contextual, sentiment-aware responses to user queries.

## 🎯 Project Overview

This project demonstrates how to build a sophisticated question-answering system that combines:
- **YouTube video transcripts** (extracted using Whisper)
- **YouTube comments** (retrieved via YouTube Data API)
- **OpenAI GPT models** for response generation
- **Pinecone vector database** for efficient document retrieval
- **Sentiment analysis** for nuanced understanding of public opinion

## 🚀 Key Features

- **Multi-modal Content Integration**: Combines video transcripts and user comments for comprehensive context
- **Sentiment Analysis**: Analyzes comment sentiment to provide emotion-aware responses
- **Advanced Retrieval**: Uses Pinecone for efficient vector-based document retrieval
- **Real-time Processing**: Transcribes videos using OpenAI Whisper
- **Evaluation Framework**: Includes comprehensive testing using Giskard for RAG system evaluation
- **Interactive Querying**: Allows users to ask specific questions about video content and community sentiment

## 🛠️ Technology Stack

- **Language Models**: OpenAI GPT-3.5 Turbo, GPT-4
- **Vector Database**: Pinecone
- **Transcription**: OpenAI Whisper
- **Video Processing**: PyTube
- **Framework**: LangChain
- **Evaluation**: Giskard RAG evaluation toolkit
- **Sentiment Analysis**: TextBlob, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- YouTube Data API key
- Pinecone API key
- Google Colab (recommended) or local environment

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-youtube-system.git
cd rag-youtube-system
```

2. Install required packages:
```bash
pip install pytube3 whisper-openai nltk
pip install --upgrade pytube
pip install langchain-openai langchain-core langchain
pip install -U langchain-pinecone
pip install -U --quiet langchain_experimental langchain_community docarray pydantic==1.10.8
pip install giskard[llm]
```

3. Set up API keys:
```python
OPENAI_API_KEY = "your-openai-api-key"
YOUTUBE_API_KEY = "your-youtube-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
```

## 📖 Usage

### Basic Setup

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import whisper

# Initialize components
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
whisper_model = whisper.load_model("base")
```

### Processing YouTube Videos

```python
# Define YouTube video URLs
YOUTUBE_VIDEOS = [
    "https://www.youtube.com/watch?v=your-video-id-1",
    "https://www.youtube.com/watch?v=your-video-id-2"
]

# Extract comments and transcripts
comments_df = get_comments(video_id)
transcription = transcribe_video(video_url)
```

### Running the RAG System

```python
# Create vector store
pinecone_store = embed_and_upload_to_pinecone(
    transcription, 
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    "your-index-name"
)

# Query the system
response = baseline_chain.invoke("What are the main topics discussed in this video?")
print(response)
```

## 🔍 Key Components

### 1. Data Collection
- **YouTube Comments**: Extracted using YouTube Data API v3
- **Video Transcripts**: Generated using OpenAI Whisper
- **Sentiment Analysis**: Applied to comments using TextBlob

### 2. Document Processing
- **Text Splitting**: Uses RecursiveCharacterTextSplitter for optimal chunking
- **Embedding Generation**: OpenAI embeddings for semantic search
- **Vector Storage**: Pinecone for scalable retrieval

### 3. RAG Pipeline
- **Retrieval**: Semantic search through video content and comments
- **Generation**: Context-aware responses using GPT models
- **Evaluation**: Comprehensive testing with Giskard framework

## 📊 Evaluation Results

The system achieves:
- **Overall Correctness**: 40%
- **Retriever Performance**: 87.5%
- **Generator Performance**: 46.67%
- **Knowledge Base**: 85.71%

### Performance by Question Type
- **Simple Questions**: 75% accuracy
- **Complex Questions**: 25% accuracy
- **Distracting Elements**: 100% accuracy

## 🎯 Use Cases

1. **Educational Content Analysis**: Extract key insights from educational videos
2. **Market Research**: Analyze public sentiment on product launches
3. **Content Summarization**: Generate concise summaries of long-form content
4. **Community Insights**: Understand audience reactions and feedback
5. **Customer Support**: Answer questions about video content automatically

## 🚧 Limitations and Future Work

### Current Limitations
- Processing time for long videos (45-60 minutes for 1-hour content)
- Limited context window for very long transcripts
- Dependency on API availability and rate limits

### Future Improvements
- **Advanced NLP Models**: Integration of domain-specific models
- **Multi-language Support**: Support for non-English content
- **Real-time Processing**: Streaming capabilities for live content
- **Enhanced Evaluation**: More comprehensive testing frameworks
- **Extended Context Sources**: Integration with forums, reviews, and social media

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Authors**: Suyash Pasari & Shreyas Prabhudev
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/rag-youtube-system

## 🙏 Acknowledgments

- OpenAI for providing GPT models and Whisper
- Pinecone for vector database capabilities
- LangChain for RAG framework
- Giskard for evaluation toolkit
- YouTube Data API for comment access

## 📚 References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Giskard RAG Evaluation](https://docs.giskard.ai/)

---

**Note**: This project is for educational and research purposes. Please ensure compliance with YouTube's Terms of Service and API usage policies.
