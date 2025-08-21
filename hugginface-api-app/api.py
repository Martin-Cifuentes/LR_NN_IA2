from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"meta-llama/Llama-2-7b-chat-hf"
token = "" #Poner Token

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32,
    token=token
)

def generar_respuesta(celebridad, pregunta):
    messages = [
        {"role": "system", "content": f"Eres {celebridad}. Responde como si fueras esa persona."},#Cambiar query...
        {"role": "user", "content": pregunta}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
    attention_mask = torch.ones_like(input_ids)
    #print(input_ids)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in respuesta:
        return respuesta.split("<|assistant|>")[-1].strip()
    return respuesta.strip()

if __name__ == "__main__":
    celebridad = input("¿Qué celebridad quieres emular? (Ej: Lionel Messi): ")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            break
        respuesta = generar_respuesta(celebridad, pregunta)
        print(f"{celebridad}: {respuesta}")