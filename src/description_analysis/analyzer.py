import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.description_analysis.policy import DEFAULT_POLICY, validate_and_sanitize
from sqlalchemy.orm import Session
from src.models.instances import get_engine  
from src.models.orm_models import LLMTraining, DimDescription
from src.models.schemas import LLMTrainingSchema, DimDescriptionSchema

engine = get_engine() # used for DB logging

def initialize_nlp_pipeline(model_name):
    
    """
    Task 1: Initialize the Tokenizer and Model
    """
    print(f"--- Initializing {model_name} Pipeline ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)   
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    return tokenizer, model

def moderate_content(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, description: str):
    """
    Task 2: Tokenize and Perform Inference
    """
    print("\n--- Analyzing Description ---")
    
    # prepare inputs
    messages = [
        {"role": "user", "content": DEFAULT_POLICY + "\n\nDescription: " + description}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    model.eval()
    
    # Perform the forward pass without tracking gradients
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
        )

    # Drop the input prompt from model output
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\nModel Output:\n", response)

    # Extract final label
    if "UNSAFE" in response.upper():
        decision = "unsafe"
    else:
        decision = "safe"


    # -----------------------------
    # AUDIT DB LOGGING
    # -----------------------------
    try:
        with Session(engine) as session:
            # pydantic validation
            desc_schema = DimDescriptionSchema(
                text=description,
                is_safe_content=(True if decision == "safe" else False)
            )
            
            desc_row = DimDescription(**desc_schema.model_dump())
            session.add(desc_row)
            session.flush()
            desc_id = desc_row.description_id
            
            # pydantic validation
            llm_schema = LLMTrainingSchema(
                reasoning=response,
                moderation_decision=decision,
                is_correct=None,
                description_key=desc_id
            )
            
            llm_row = LLMTraining(**llm_schema.model_dump())
            session.add(llm_row)
            session.commit()
    except Exception as e:
        print("DB logging failed:", e)
        
    print("\nFinal Moderation Decision:", decision)
    return decision

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    my_tokenizer, my_model = initialize_nlp_pipeline(MODEL_NAME)
    
    description_text = "HerSpace is so cool. Love this app." # input a post description here!

    clean_text = validate_and_sanitize(description_text)
    
    if my_tokenizer and my_model:
        moderation_decision = moderate_content(my_tokenizer, my_model, clean_text)
