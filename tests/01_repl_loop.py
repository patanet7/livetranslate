from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from collections import deque

# Ring buffer for last 4 user inputs
recent_inputs = deque(maxlen=4)
# Initialize tokenizer and model
model_name = "Qwen/Qwen3-14B-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Sampling settings for non-thinking (fast generation)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=300,
    presence_penalty=1.5,  
    stop_token_ids=[tokenizer.eos_token_id],
)

# Load and cache model
llm = LLM(
    model=model_name,
    quantization="awq_marlin",
    dtype="half",
    tensor_parallel_size=1,
    max_model_len=512,
)

# Define system prompt once
SYSTEM_PROMPT = (
    "You are an expert bilingual translator who fluently translates between Chinese and English.\n"
    "If the input is in Chinese characters (Hanzi), translate to English.\n"
    "If the input is in English, translate to Chinese.\n"
    "Avoid explanations. Only return the translated sentence with no formatting.\n"
    "Translate the following text:\n"
)

def get_prompt(input_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


# Ring buffer for last 4 user inputs
recent_inputs = deque(maxlen=4)

def translate_text(input_text: str) -> str:
    # Your existing translation logic here
    prompt = f"{SYSTEM_PROMPT}{input_text}\nTranslation:"
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()

def main():
    print("🌐 Sentence Translator with Memory (last 4 inputs)")
    print("Type a sentence in Chinese or English. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Input > ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            # Save to history
            recent_inputs.append(user_input)

            # Perform translation
            translated = translate_text(user_input)
            print(f"Translation > {translated}\n")

            # Optional: show memory buffer
            print("Recent inputs:")
            for i, sentence in enumerate(recent_inputs, 1):
                print(f"  {i}: {sentence}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()