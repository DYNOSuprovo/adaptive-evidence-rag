import os
import operator
from typing import List, Dict, Any, Tuple, TypedDict, Annotated, Sequence
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Import existing modules
from src.retriever import EvidenceAwareRetriever

# Define State
class AgentState(TypedDict):
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
    def __init__(self, retriever_pipeline: EvidenceAwareRetriever, llm_model: str = "gpt-3.5-turbo"):
        self.retriever_pipeline = retriever_pipeline
        # We will use OpenAI for the Synthesizer LLM
        # Ensure OPENAI_API_KEY is in environment or handle it gracefully
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
            
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("retriever_agent", self.retriever_node)
        workflow.add_node("evidence_critic_agent", self.evidence_critic_node)
        workflow.add_node("utility_judge_agent", self.utility_judge_node)
        workflow.add_node("stability_auditor_agent", self.stability_auditor_node)
        workflow.add_node("synthesizer_agent", self.synthesizer_node)
        
        # Define Edges
        workflow.set_entry_point("retriever_agent")
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
        llm_to_use = None
        if self.llm and os.environ.get("OPENAI_API_KEY"):
            llm_to_use = self.llm
        elif getattr(self, "ollama_llm", None):
            llm_to_use = self.ollama_llm
            
        if not llm_to_use:
            return {"final_answer": "System Notice: Neither OpenAI nor local Ollama LLMs are available. The LangGraph pipeline filtered the documents successfully, but the Synthesizer Agent could not run."}
            
        question = state['question']
        docs = state['filtered_documents']
        
        if not docs:
            return {"final_answer": "I don't have enough verified evidence to answer this question."}
            
        context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(docs)])
        
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
