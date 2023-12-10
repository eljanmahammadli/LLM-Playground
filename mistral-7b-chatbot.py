from transformers import AutoTokenizer, pipeline
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    max_new_tokens=1000,
    context_length=6000,
    gpu_layers=50,
    hf=True,
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
pipe = pipeline(model=model, tokenizer=tokenizer, task="text-generation")
bos_token, eos_token, instr_bgn, instr_end = "<s>", "</s>", "[INST]", "[/INST]"

context = """Your name is Qazanfar and you are a helpful AI assistant who answers questions as accurately and concisely as possible.
You are from Beylagan city which is located in the south of Azerbaijan.
You are also very good at Python programming. You write Python code as follows:
```python
CODE_BLOCK_HERE
``` \n
"""
# context = """Recent work pre-training Transformers with self-supervised objectives on large text corpora has shown great success when fine-tuned on downstream NLP tasks including text summa- rization. However, pre-training objectives tai- lored for abstractive text summarization have not been explored. Furthermore there is a lack of systematic evaluation across diverse do- mains. In this work, we propose pre-training large Transformer-based encoder-decoder mod- els on massive text corpora with a new self- supervised objective. In PEGASUS, important sentences are removed/masked from an input doc- ument and are generated together as one output sequence from the remaining sentences, similar to an extractive summary. We evaluated our best PEGASUS model on 12 downstream summariza- tion tasks spanning news, science, stories, instruc- tions, emails, patents, and legislative bills. Experi- ments demonstrate it achieves state-of-the-art per- formance on all 12 downstream datasets measured by ROUGE scores. Our model also shows surpris- ing performance on low-resource summarization, surpassing previous state-of-the-art results on 6 datasets with only 1000 examples. Finally we validated our results using human evaluation and show that our model summaries achieve human performance on multiple datasets\n.
# """
messages = {
    # "Your name is Qazanfar and you are a helpful AI assistant who answers questions as accurately and concisely as possible. You are also very good at Python programming": "Alright, how may I assist you?",
    "Hello, how are you?": "I am fine thanks, what about you?",
    "What is 2 + 2?": "Two plus two equals to the 4.",
}
prompt = (
    bos_token  # beginning of the sentence
    + context  # preprompt instructions
    + "".join(  # few-shot prompts
        [f"{instr_bgn} {key} {instr_end}\n{value}{eos_token}\n" for key, value in messages.items()]
    )
)
print("\nMistralChatBot\n")
print(bos_token, end="")
for k, v in messages.items():  # pretty print the predefined messages
    print(f"{instr_bgn} {k} {instr_end}\n[ANSW] {v}{eos_token}\n")

while True:
    inp = input(f"{instr_bgn} ")
    if inp == "exit":  # exit the chatbot
        break
    usr_prompt = f"{instr_bgn} {inp} {instr_end}\n"
    prompt += usr_prompt
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)  # TODO: why False?
    generated_ids = model.generate(**encodeds, max_new_tokens=5000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    response = decoded[0].split(instr_end)[-1].strip()  # pluck last model response
    prompt += f"{response}\n"
    print(f"[ANSW] {response}\n")

    # run python code via LLM. BE CAREFUL though LOL :D
    if "```python" in response and input("Do you want to run the Python code? 'y' or 'n': ") == "y":
        code = response.split("```python")[1].split("```")[0]
        try:
            exec(code)
        except Exception as exc:
            print(f"Dumb model, wrote bad code: {exc}")
