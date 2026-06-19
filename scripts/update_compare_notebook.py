import json

def update_notebook():
    path = "notebooks/compare_rag_demo.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # Create the new cells to append
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Generator Model Comparison\n",
                "\n",
                "Now that we have filtered down the context to a single highly useful and independent document, let's see how different LLMs synthesize it.\n",
                "We will compare:\n",
                "1. **Baseline Ollama Model:** Standard `phi3` or `llama3` running locally.\n",
                "2. **Base Pre-trained Model:** `Qwen/Qwen1.5-0.5B-Chat` without any fine-tuning.\n",
                "3. **Our Custom Fine-Tuned Model:** `Qwen1.5-0.5B-Evidence-RAG` (using our LoRA adapters)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_ollama import ChatOllama\n",
                "from langchain_core.messages import HumanMessage\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "from peft import PeftModel\n",
                "import torch\n",
                "\n",
                "question = \"Who invented the transformer architecture?\"\n",
                "docs = result.filtered_documents\n",
                "context = \"\\n\\n\".join([f\"[Doc {i+1}] {doc}\" for i, doc in enumerate(docs)])\n",
                "\n",
                "prompt = f'''You are an advanced Evidence-Aware AI Assistant. \\n\"\n",
                "Answer the following user question using ONLY the provided verified evidence.\\n\"\n",
                "If the evidence does not contain the answer, say \"I don't have enough verified evidence to answer this.\"\\n\\n\"\n",
                "Evidence:\\n\"\n",
                "{context}\\n\\n\"\n",
                "Question:\\n\"\n",
                "{question}\\n\\n\"\n",
                "Answer:'''\n",
                "\n",
                "print(\"========== GENERATOR COMPARISON ==========\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Baseline Ollama\n",
                "try:\n",
                "    ollama = ChatOllama(model=\"phi3\", temperature=0.0)\n",
                "    ollama_res = ollama.invoke([HumanMessage(content=prompt)])\n",
                "    print(f\"\\n[1] Baseline Ollama (phi3):\\n{ollama_res.content}\")\n",
                "except Exception as e:\n",
                "    print(f\"\\n[1] Baseline Ollama (phi3):\\nError or not running - {e}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Base Pre-trained Model (Qwen-0.5B)\n",
                "try:\n",
                "    model_name = \"Qwen/Qwen1.5-0.5B-Chat\"\n",
                "    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n",
                "    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=\"auto\", torch_dtype=torch.float16)\n",
                "    \n",
                "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "    outputs = base_model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)\n",
                "    base_res = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)\n",
                "    print(f\"\\n[2] Base Model (Qwen-0.5B):\\n{base_res}\")\n",
                "except Exception as e:\n",
                "    print(f\"\\n[2] Base Model (Qwen-0.5B):\\nError loading base model - {e}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Our Fine-Tuned Model (Qwen-0.5B + LoRA)\n",
                "lora_path = \"../models/qwen-evidence-rag-lora-final\"\n",
                "try:\n",
                "    finetuned_model = PeftModel.from_pretrained(base_model, lora_path)\n",
                "    outputs_ft = finetuned_model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)\n",
                "    ft_res = tokenizer.decode(outputs_ft[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)\n",
                "    print(f\"\\n[3] Our Fine-Tuned Model:\\n{ft_res}\")\n",
                "except Exception as e:\n",
                "    print(f\"\\n[3] Our Fine-Tuned Model:\\nNot trained yet! Run finetune_synthesizer.ipynb first. Error: {e}\")\n"
            ]
        }
    ]
    
    nb["cells"].extend(new_cells)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook()
