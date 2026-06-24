import os
import operator
from typing import List, Dict, Any, Tuple, TypedDict, Annotated, Sequence
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Import existing modules
from src.retriever import EvidenceAwareRetriever

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Define State
class AgentState(TypedDict):
    intent: str
    question: str
    query_used: str
    original_documents: List[str]
    filtered_documents: List[str]
    independence_score: float
    utility_score: float
    stability_score: float
    overall_quality: float
    final_answer: str
    metadata: Dict[str, Any]
    retry_count: int

class RAGMultiAgentSystem:
        try:
            self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        except Exception as e:
            print(f"Warning: Failed to init ChatOpenAI, maybe missing API key? {e}")
            self.llm = None
            
        try:
            self.ollama_llm = ChatOllama(model="phi3", temperature=0.2)
        except Exception as e:
            print(f"Warning: Failed to init ChatOllama: {e}")
            self.ollama_llm = None
            
        self.use_local_qwen = True
        self.qwen_model = None
        self.qwen_tokenizer = None
        if self.use_local_qwen:
            try:
                print("[AgentSystem] Loading local Qwen fine-tuned model (4-bit)...")
                model_name = "Qwen/Qwen1.5-0.5B-Chat"
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                # Check if the LoRA model directory exists before loading
                lora_path = "models/qwen-evidence-rag-lora-final"
                if os.path.exists(lora_path):
                    self.qwen_model = PeftModel.from_pretrained(base_model, lora_path)
                    print("[AgentSystem] Local Qwen + LoRA loaded successfully.")
                else:
                    print(f"[AgentSystem] Warning: LoRA path {lora_path} not found. Using base model.")
                    self.qwen_model = base_model
            except Exception as e:
                print(f"Warning: Failed to load local Qwen model: {e}")
            
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("router_agent", self.router_node)
        workflow.add_node("direct_answer_agent", self.direct_answer_node)
        workflow.add_node("retriever_agent", self.retriever_node)
        workflow.add_node("evidence_critic_agent", self.evidence_critic_node)
        workflow.add_node("utility_judge_agent", self.utility_judge_node)
        workflow.add_node("stability_auditor_agent", self.stability_auditor_node)
        workflow.add_node("synthesizer_agent", self.synthesizer_node)
        
        # Define Edges
        workflow.set_entry_point("router_agent")
        
        workflow.add_conditional_edges(
            "router_agent",
            lambda state: "knowledge" if state.get("intent") == "KNOWLEDGE" else "direct",
            {
                "knowledge": "retriever_agent",
                "direct": "direct_answer_agent"
            }
        )
        
        workflow.add_edge("direct_answer_agent", END)
        
        workflow.add_edge("retriever_agent", "evidence_critic_agent")
        workflow.add_edge("evidence_critic_agent", "utility_judge_agent")
        
        workflow.add_conditional_edges(
            "utility_judge_agent",
            self.route_after_utility,
            {
                "retry": "retriever_agent",
                "continue": "stability_auditor_agent"
            }
        )
        
        workflow.add_edge("stability_auditor_agent", "synthesizer_agent")
        workflow.add_edge("synthesizer_agent", END)
        
        return workflow.compile()
        
    def route_after_utility(self, state: AgentState):
        """Route to synthesizer or retry if all documents are filtered."""
        if not state.get('filtered_documents') and state.get('retry_count', 0) < 1:
            return "retry"
        return "continue"

    def _generate_with_local_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to generate text using the local Qwen model."""
        if not (self.qwen_model and self.qwen_tokenizer):
            raise ValueError("Local Qwen model is not loaded.")
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.qwen_model.device)
        
        generated_ids = self.qwen_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def router_node(self, state: AgentState):
        """Classifies the query intent to route properly."""
        question = state['question']
        
        prompt = f'''Analyze the user's input and classify its intent into EXACTLY ONE of the following categories:
1. "MATH": Mathematical calculations, equations, or logic puzzles.
2. "CONVERSATIONAL": Greetings, chitchat, or direct requests that don't require external facts.
3. "KNOWLEDGE": Factual questions requiring Wikipedia or external document search.

User Input: {question}

Respond with ONLY the category name (MATH, CONVERSATIONAL, or KNOWLEDGE).'''

        intent = "KNOWLEDGE"
        try:
            if self.qwen_model:
                system_prompt = "You are an intelligent query router. Respond only with the exact category name requested."
                response_text = self._generate_with_local_llm(system_prompt, prompt)
            elif self.llm:
                response_text = self.llm.invoke([HumanMessage(content=prompt)]).content
            else:
                response_text = "KNOWLEDGE"
                
            intent_raw = response_text.strip().upper()
            if "MATH" in intent_raw: intent = "MATH"
            elif "CONVERSATIONAL" in intent_raw: intent = "CONVERSATIONAL"
            else: intent = "KNOWLEDGE"
        except Exception as e:
            print(f"Router error: {e}")
            intent = "KNOWLEDGE"
            
        metadata = state.get("metadata", {})
        metadata["intent"] = intent
        return {"intent": intent, "metadata": metadata}
        
    def direct_answer_node(self, state: AgentState):
        """Answers math or conversational queries directly without RAG."""
        question = state['question']
        
        try:
            if self.qwen_model:
                system_prompt = "You are a helpful AI assistant. Answer the user's question directly."
                user_prompt = f"Question: {question}"
                response_text = self._generate_with_local_llm(system_prompt, user_prompt)
            elif self.llm:
                prompt = f"Answer the following directly and accurately:\n\nUser: {question}\n\nAnswer:"
                response_text = self.llm.invoke([HumanMessage(content=prompt)]).content
            else:
                response_text = "System Notice: No local or external LLM available to answer."
            
            return {"final_answer": response_text}
        except Exception as e:
            return {"final_answer": f"Error generating answer: {str(e)}"}
        
    def retriever_node(self, state: AgentState):
        """Generates variants and retrieves initial candidate documents."""
        question = state['question']
        retry_count = state.get('retry_count', 0)
        
        # If retrying, expand query
        if retry_count > 0:
            question = question + " overview summary details"
            
        # Use existing retriever's search policy
        if self.retriever_pipeline.use_search_policy:
            query_used, retrieved = self.retriever_pipeline.retrieve_with_policy(question)
        else:
            query_used = question
            retrieved = self.retriever_pipeline.retrieve(question)
            
        original_docs = [r[2] for r in retrieved]
        
        relevance_dict = {}
        # Apply reranker to calculate relevance logits, DO NOT drop <= 0, let Utility Sigmoid handle it
        if original_docs and getattr(self.retriever_pipeline, "reranker", None) is not None:
            reranked = self.retriever_pipeline.rerank(question, original_docs)
            for idx, score in reranked:
                relevance_dict[original_docs[idx]] = score
            original_docs = [original_docs[idx] for idx, score in reranked]
            
        metadata = state.get("metadata", {})
        metadata["num_original"] = len(retrieved)
        metadata["relevance_dict"] = relevance_dict
            
        return {
            "query_used": query_used,
            "original_documents": original_docs,
            "filtered_documents": original_docs, # passed to next node
            "metadata": metadata,
            "retry_count": retry_count + 1
        }
        
    def evidence_critic_node(self, state: AgentState):
        """Filters redundant docs using Independence Scorer."""
        documents = state['filtered_documents']
        if not documents:
            return {"independence_score": 0.0}
            
        doc_embeddings = self.retriever_pipeline.embedder.encode(documents, convert_to_numpy=True)
        
        # Score independence
        independence_results = self.retriever_pipeline.independence_scorer.score_independence(
            documents, 
            embeddings=doc_embeddings
        )
        independence_aggregate = self.retriever_pipeline.independence_scorer.get_aggregate_score(
            independence_results
        )
        
        # Filter
        filtered_docs, keep_indices = self.retriever_pipeline.independence_scorer.filter_redundant_documents(
            documents,
            results=independence_results
        )
        
        metadata = state.get("metadata", {})
        metadata["independence_aggregate"] = independence_aggregate
        
        return {
            "filtered_documents": filtered_docs,
            "independence_score": independence_aggregate.get('mean_independence', 0.5),
            "metadata": metadata
        }
        
    def utility_judge_node(self, state: AgentState):
        """Scores utility and filters out useless docs."""
        question = state['question']
        documents = state['filtered_documents']
        
        if not documents:
            return {"utility_score": 0.0}
            
        metadata = state.get("metadata", {})
        relevance_dict = metadata.get("relevance_dict", {})
        relevance_logits = [relevance_dict.get(doc, 0.0) for doc in documents]
            
        utility_results = self.retriever_pipeline.utility_scorer.score_utility(
            question, documents, relevance_logits=relevance_logits
        )
        utility_aggregate = self.retriever_pipeline.utility_scorer.get_aggregate_metrics(utility_results)
        
        filtered_docs, keep_indices = self.retriever_pipeline.utility_scorer.filter_by_utility(
            documents,
            results=utility_results,
            top_k=min(5, len(documents)),
            threshold=self.retriever_pipeline.utility_scorer.min_utility_threshold
        )
        
        metadata["utility_aggregate"] = utility_aggregate
        
        return {
            "filtered_documents": filtered_docs,
            "utility_score": utility_aggregate.get('mean_utility', 0.5),
            "metadata": metadata
        }
        
    def stability_auditor_node(self, state: AgentState):
        """Checks behavioral stability."""
        stability_score = 0.5 # Placeholder, real stability check is too slow for sync requests
        
        indep_score = state.get('independence_score', 0)
        util_score = state.get('utility_score', 0)
        
        overall_quality = (
            0.35 * indep_score +
            0.35 * util_score +
            0.30 * stability_score
        )
        
        metadata = state.get("metadata", {})
        metadata["num_filtered"] = len(state['filtered_documents'])
        
        return {
            "stability_score": stability_score,
            "overall_quality": overall_quality,
            "metadata": metadata
        }
        
    def synthesizer_node(self, state: AgentState):
        """Generates final answer using LLM."""
        question = state['question']
        docs = state['filtered_documents']
        
        if not docs:
            return {"final_answer": "I don't have enough verified evidence to answer this question."}
            
        context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(docs)])
        
        # Priority 1: Use local fine-tuned Qwen model
        if getattr(self, "use_local_qwen", False) and getattr(self, "qwen_model", None):
            try:
                system_prompt = "You are an advanced Evidence-Aware AI Assistant. Answer the user's question using ONLY the provided verified evidence. If the evidence does not contain the answer, say 'I don't have enough verified evidence to answer this.'"
                user_prompt = f"Evidence:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                
                response = self._generate_with_local_llm(system_prompt, user_prompt)
                return {"final_answer": response}
            except Exception as e:
                print(f"Error generating answer with local Qwen: {e}")
                # Fallback to other LLMs if it fails
                
        llm_to_use = getattr(self, "llm", None) or getattr(self, "ollama_llm", None)
            
        if not llm_to_use:
            return {"final_answer": "System Notice: Local Qwen and External LLMs are unavailable. The LangGraph pipeline filtered the documents successfully, but the Synthesizer Agent could not run."}
            
        prompt = f'''You are an advanced Evidence-Aware AI Assistant. 
Answer the following user question using ONLY the provided verified evidence.
If the evidence does not contain the answer, say "I don't have enough verified evidence to answer this."

Evidence:
{context}

Question:
{question}

Answer:'''

        try:
            response = llm_to_use.invoke([HumanMessage(content=prompt)])
            return {"final_answer": response.content}
        except Exception as e:
            return {"final_answer": f"Error generating answer: {str(e)}"}
        
    def run(self, question: str) -> dict:
        """Run the multi-agent workflow."""
        initial_state = {
            "intent": "",
            "question": question,
            "query_used": "",
            "original_documents": [],
            "filtered_documents": [],
            "independence_score": 0.0,
            "utility_score": 0.0,
            "stability_score": 0.0,
            "overall_quality": 0.0,
            "final_answer": "",
            "metadata": {},
            "retry_count": 0
        }
        
        final_state = self.workflow.invoke(initial_state)
        return final_state
