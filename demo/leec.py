import gradio as gr
import requests
import json
from transformers import LlamaForCausalLM, LlamaTokenizer, TextIteratorStreamer
import torch
import argparse
from tqdm import tqdm
output_path="lawyer_llama_output.json"
MAXLEN=2048


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/root/autodl-tmp/lawyer-llama-13b-beta1.0")
    parser.add_argument("--classifier_url", type=str, default="")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_chat_mode", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    classifier_url = args.classifier_url

    print("Loading model...")
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    if args.load_in_8bit:
        model = LlamaForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)
    else:
        model = LlamaForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
    print("Model loaded.")

    if args.use_chat_mode:
        print("Chat mode enabled.")
        print("If you want to start a new chat, enter 'clear' please")
        chat_history = []
    #2048
    with open("../data/leec_4163去重_0209.json", "r", encoding="UTF-8") as f:
        data= json.load(f)
    output_list=[]
    Count=0
    pbar = tqdm(total=100)
    pbar.update(29)#6)
    for qa_data in data[29:101]:
        qa_data["instruction"]="你是人工智能法律助手“Lawyer LLaMA”，能够回答与中国法律相关的问题。\n### Human: "+qa_data["instruction"]
        content = qa_data["instruction"]+qa_data["input"]+"\n### Assistant: "
        if len(content) > MAXLEN:
            input_list = qa_data["input"].split("。")
            input = ""
            for sen in input_list:
                if len(qa_data["instruction"]+ input + sen + "。\n### Assistant: ") < MAXLEN:
                    input += sen + "。"
                else:
                    break
            content = qa_data["instruction"]+input+"\n### Assistant: "
            Count += 1
            # for history_pair in chat_history:
            #     input_text += f"### Human: {history_pair[0]}\n### Assistant: {history_pair[1]}\n"
            #input_text += f"### Human: {current_user_input}\n### Assistant: "

        input_ids = tokenizer(content, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=2341, do_sample=False, repetition_penalty=1.1)
        output_text = str(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # skip prompt
        output_text = output_text[len(content):]
        output_list.append({"ansewer":output_text})
        print("[AI] >>> " + output_text)
        pbar.update(1)

    with open(output_path, 'w', encoding='utf-8') as b:
        # ensure_ascii 显示中文，不以ASCII的方式显示
        json.dump(output_list, b, ensure_ascii=False)  # indent 缩进

