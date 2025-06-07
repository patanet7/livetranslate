from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "TheBloke/ALMA-7B-Pretrain-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Sampling settings
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=400,
    stop_token_ids=[tokenizer.eos_token_id],
)

llm = LLM(model=model_name, quantization="awq_marlin", dtype="half", tensor_parallel_size=1)

chinese_input = '我们将尝试快速翻译此内容。'
# prepare the model input
prompt = f"Translate this from Chinese to English:\nChinese: {chinese_input}\nEnglish: "

outputs = llm.generate(prompt, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)

