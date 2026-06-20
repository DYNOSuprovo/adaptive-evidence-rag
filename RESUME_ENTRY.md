**Adaptive Evidence RAG** – *Intelligent, Self-Evaluating Retrieval Pipeline*  
*FastAPI, React.js, PyTorch, LangGraph, Qwen-4bit, Hugging Face Spaces*  
[GitHub](https://github.com/DYNOSuprovo/adaptive-evidence-rag) | [Live Demo](https://huggingface.co/spaces/Dyno1307/adaptive-evidence-rag)  

• Achieved a <10% hallucination rate and >75% Exact Match (EM) accuracy by engineering an adaptive multi-agent RAG pipeline that strictly filters low-utility context before generation.  
• Accelerated evidence processing and reduced redundant LLM token generation by ˜80% through cross-encoder NLI independence scoring (threshold ≥ 0.70), dropping duplicate documents from the context window.  
• Deployed a full-stack containerized application (React + FastAPI) to Hugging Face Spaces, orchestrating 5 parallel retrieval/scoring modules and a 4-bit quantized Qwen LLM for <2s inference latency.
