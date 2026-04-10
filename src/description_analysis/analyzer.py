import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# TODO: review default_ policy + input sanitation
DEFAULT_POLICY = """
You are an expert sentiment analysis agent for the HerSpace social media platform.

Follow these rules:
- Always use chain-of-thought reasoning with ReAct looping
    Loop to follow:
    1. Thought: describe your goal
    2. Action: define what to to to reach the goal
    3. Observation: State what your action achieved. 
- Output in in the following JSON format:
    {
        Reasoning: [ReAct loop, including thought, action, and observation]
        Decision: [final answer: "safe" or "unsafe]
    }
    
Your task:
- Analyze text descriptions and determine if the content aligns with HerSpace Policies listed below.
- Return your final answer (moderation decision) as: "safe" or "unsafe".
"unsafe" marking content that does not adhere to policies.

Policies:
1. Topics of violence, self-harm, or harassment are prohibited.
2. Post descriptions should pass the Bechdel test meaning the central purpose of the text should not be centered around a man.
    Example violation: "I love my man. He's the best husband ever. He's my whole world."
    Violation reasoning: "This description is in violation of the Bechdel test policy on HerSpace. Content on this platform should not be centered around men"
3. The following phrases are strictly prohibited from post descriptions:
    "guys night out", "boys night out", "out with the boys", "man cave", "my boysss", "alpha male", "sigma male"
4. Profanity is strictly prohibited.
    
Example content moderation:
    Description 1: "Happy at work! My coworker, Charles, helped me out today.
    Decision 1: "safe"
    Reasoning: "A man, Charles, is mentioned, but he is not the central focus of the text, the user's work is the focus."
    
    Description 2: "Bachelor weekend!! "Night out with my guyss"
    Decision 2: "unsafe"
    Reasoning: "Description uses variations of prohibited words: "guys night out", "my boysss."
"""

SIMPLE_POLICY = """
You are an expert sentiment analysis agent for the HerSpace social media platform.

Your task:
- Analyze text descriptions and determine if the content aligns with HerSpace Policies listed below.
- Return your final answer (moderation decision) as: "safe" or "unsafe".
"unsafe" marking content that does not adhere to policies.

Policies:
1. Topics of violence, self-harm, or harassment are prohibited.
2. Post descriptions should pass the Bechdel test meaning the central purpose of the text should not be centered around a man.
    Example violation: "I love my man. He's the best husband ever. He's my whole world."
    Violation reasoning: "This description is in violation of the Bechdel test policy on HerSpace. Content on this platform should not be centered around men"
3. The following phrases are strictly prohibited from post descriptions:
    "guys night out", "boys night out", "out with the boys", "man cave", "my boysss", "alpha male", "sigma male"
4. Profanity is strictly prohibited.
    
Example content moderation:
    Description 1: "Happy at work! My coworker, Charles, helped me out today.
    Decision 1: "safe"
    
    Description 2: "Bachelor weekend!! "Night out with my guyss"
    Decision 2: "unsafe"
"""

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

def analyze_sentiment(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, description: str):
    """
    Task 2: Tokenize and Perform Inference
    """
    print("\n--- Analyzing Description ---")
    
    # prepare inputs
    messages = [
        # {"role": "system", "content": DEFAULT_POLICY},
        {"role": "user", "content": SIMPLE_POLICY + "\n\nDescription: " + description}
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
            max_new_tokens=100,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nModel Output:\n", response)

    # Extract final label
    if "UNSAFE" in response.upper():
        decision = "unsafe"
    else:
        decision = "safe"


    print("\nFinal Moderation Decision:", decision)
    return decision

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    
    my_tokenizer, my_model = initialize_nlp_pipeline(MODEL_NAME)
    
    sample_description = "guys night out is epic"
    
    if my_tokenizer and my_model:
        analyze_sentiment(my_tokenizer, my_model, sample_description)
