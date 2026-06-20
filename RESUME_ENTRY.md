**Adaptive Evidence RAG** – *Intelligent, Self-Evaluating Retrieval Pipeline*  
*FastAPI, React.js, PyTorch, LangGraph, Qwen-4bit, Hugging Face Spaces*  
[GitHub](https://github.com/DYNOSuprovo/adaptive-evidence-rag) | [Live Demo](https://huggingface.co/spaces/Dyno1307/adaptive-evidence-rag)  

• Engineered an adaptive multi-agent RAG architecture that autonomously evaluates retrieved evidence for independence and utility, drastically reducing hallucination rates by filtering out redundant or unverified context.  
• Implemented a multi-stage LangGraph pipeline featuring an Auditor Agent and a dynamic Search Policy, enabling the system to safely reject low-utility documents rather than forcing fabricated LLM responses.  
• Deployed a robust full-stack application (React + FastAPI) to Hugging Face Spaces, orchestrating parallelized cross-encoder scoring modules and a 4-bit quantized Qwen LLM for efficient, low-latency inference.
