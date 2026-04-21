"""Streamlit UI — model load matches eval/gen_predictions_qwen.py; prompt uses configurable system text."""

import gc
import importlib.util
from pathlib import Path

import streamlit as st
import torch

from train.qwen_lora_finetune import QwenLoraConfig

_gpq_path = Path(__file__).resolve().parent / "eval" / "gen_predictions_qwen.py"
_spec = importlib.util.spec_from_file_location("gen_predictions_qwen", _gpq_path)
_gpq = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gpq)
load_model = _gpq.load_model
format_sql_response = _gpq.format_sql_response

MAX_NEW_TOKENS = 128


@st.cache_resource
def load_resource(model_name: str, adapter_path: str):
    ap = adapter_path.strip() if adapter_path and adapter_path.strip() else None
    return load_model(model_name, ap)


def release_model():
    """Drop all references to the current model and free GPU memory."""
    if "model" in st.session_state:
        del st.session_state["model"]
    if "tokenizer" in st.session_state:
        del st.session_state["tokenizer"]
    load_resource.clear()
    gc.collect()
    torch.cuda.empty_cache()


st.set_page_config(page_title="18786 Project: Text-to-SQL", layout="wide")

with st.sidebar:
    st.title("Settings")
    model_path = st.text_input("Base Model", QwenLoraConfig.MODEL_NAME)
    adapter_path = st.text_input("Adapter Path", "")

    if st.button("Load / Reload Model"):
        with st.spinner("Releasing previous model..."):
            release_model()
        with st.spinner("Loading model..."):
            st.session_state.active_model_path = model_path
            st.session_state.active_adapter_path = adapter_path
            m, tok = load_resource(model_path, adapter_path)
            st.session_state.model = m
            st.session_state.tokenizer = tok
        st.success("Model loaded!")

    st.divider()
    system_prompt = st.text_area(
        "System Instruction",
        value=(
            "Complete sqlite SQL query only and with no explanation, "
            "and do not select extra columns that are not explicitly requested in the query."
        ),
        height=150,
    )

    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

st.title("18786 Project: Text-to-SQL Interface")

with st.expander("Schema Definition", expanded=True):
    schema_context = st.text_area(
        "Database Schema",
        placeholder="Input table definitions and foreign keys here (same role as db_info in eval)...",
        height=200,
    )

model_loaded = "model" in st.session_state

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["content"], language="sql")
        else:
            st.write(msg["content"])

if not model_loaded:
    st.info("Load a model from the sidebar to get started.")
elif user_input := st.chat_input("Enter natural language query..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    formatted_prompt = (
        f"{system_prompt}\n"
        f"### Sqlite SQL tables, with their properties:\n"
        f"#\n{schema_context}\n#\n"
        f"### {user_input}\n"
        "SELECT"
    )

    with st.chat_message("assistant"):
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer

        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        final_sql = format_sql_response(raw)

        st.code(final_sql, language="sql")
        st.session_state.messages.append({"role": "assistant", "content": final_sql})