import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from qwen_lora_finetune import QwenLoraConfig

@st.cache_resource
def load_resource(model_name, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=dtype, 
        trust_remote_code=True
    )
    
    if adapter_path and adapter_path.strip():
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="18786 Project: Text-to-SQL", layout="wide")

with st.sidebar:
    st.title("Settings")
    model_path = st.text_input("Base Model", QwenLoraConfig.MODEL_NAME)
    adapter_path = st.text_input("Adapter Path", "")
    
    st.divider()
    system_prompt = st.text_area(
        "System Instruction", 
        value="Complete sqlite SQL query only and with no explanation, and do not select extra columns that are not explicitly requested in the query.",
        height=150
    )
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

st.title("18786 Project: Text-to-SQL Interface")

with st.expander("Schema Definition", expanded=True):
    schema_context = st.text_area(
        "Database Schema",
        placeholder="Input table definitions and foreign keys here...",
        height=200
    )

tokenizer, model = load_resource(model_path, adapter_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="sql")
        else:
            st.write(msg["content"])

if user_input := st.chat_input("Enter natural language query..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    formatted_prompt = (
        f"{system_prompt}\n"
        f"### Sqlite SQL tables, with their properties:\n"
        f"#\n{schema_context}\n#\n"
        f"### {user_input}\n"
        f"SELECT"
    )

    with st.chat_message("assistant"):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        final_sql = "SELECT " + generated_text
        
        st.code(final_sql, language="sql")
        st.session_state.messages.append({"role": "assistant", "content": final_sql})