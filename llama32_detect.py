# Dir Related
import os
from pathlib import Path
import pandas as pd

import requests
import torch
from PIL import Image
from IPython.display import display, HTML
from transformers import MllamaForConditionalGeneration, AutoProcessor

#####

def msgs(prompt, with_image=True):
    if with_image:
        return [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    else:
        return [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
    
def img2text(input_path, output_file = None, exportedfile_indexing = False, show_img = False, max_new_tokens = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #device_map="auto",
    )
    
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    #tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct', trust_remote_code=True)
    #model.eval()
    dir = [input_path]
    if os.path.isdir(input_path):
        dir = os.listdir(input_path)
    prompt_orig = """You are a medical image analysis expert. Your task is to carefully analyze the image and determine if it shows a patient being assisted in turning by another person. Here are some examples:

Example 1:
Image: A nurse standing next to a hospital bed with her hands on a patient's shoulder and hip, clearly in the process of rolling them from their back to their side.
Analysis: True - This shows active turning assistance because:
- Direct physical contact between caregiver and patient
- Clear repositioning movement from back to side
- Proper supportive hand placement for turning

Example 2:
Image: A patient lying still in bed while a nurse stands nearby checking IV fluids.
Analysis: False - This is not turning assistance because:
- No physical contact for movement support
- Patient position is static
- Caregiver is performing different care tasks

Example 3:
Image: A caregiver with hands positioned near a patient's shoulders, standing in a stance ready to assist movement.
Analysis: True - While turning hasn't started, it's imminent because:
- Caregiver positioning indicates preparation for movement
- Hands are positioned appropriately for turning support
- Stance shows readiness to assist

Now analyze the given image considering:

1. People Present
- Is there a patient visible?
- Is there at least one caregiver/assistant visible?
- What is their relative positioning?

2. Physical Contact & Assistance
- Is there direct physical contact between caregiver and patient?
- Where and how is the contact being made (hands, arms, etc.)?
- Is the caregiver in a stance that indicates they are providing support?

3. Patient Position & Movement
- What is the patient's current position?
- Is there evidence of ongoing movement or repositioning?
- What appears to be the intended direction of movement?

4. Level of Assistance
- How actively is the caregiver supporting the movement?
- What specific actions show they are helping with turning?
- Is this clearly a turning assistance scenario?

Based on your analysis, provide:
1. A detailed explanation of what you observe
2. Your final determination: True if turning assistance is occurring/imminent, False if not
3. The key evidence that led to your conclusion
"""
#Remember: Even if turning hasn't started but is clearly about to occur (caregiver positioned and ready to assist), this should be classified as TRUE.


    
    #question = 'What is in the image?'
    #msgs = [{'role': 'user', 'content': prompt}]
    data = []
    result = {}
    for i, image_path in enumerate(sorted(dir)):
        # Read the image
        if os.path.isdir(input_path):
            image = Image.open(Path(input_path).joinpath(image_path))
        else:
            image = Image.open(image_path)
        
        
        # Describe the image
        input_text = processor.apply_chat_template(msgs("Describe the image in detail."), add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0]).split("<|end_header_id|>")[-1].replace('\n', ' ')
        
        # Show the steps based on the image
        prompt = "The picture is about the following:\n" +res +'\n' + prompt_orig
        
        input_text = processor.apply_chat_template(msgs(prompt), add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0])
        
        
        print('\n', i, image_path)
        #print(generated_text,'\n')
        print('Full Response\n', res)
        reason = res.split("<|end_header_id|>")[-1]
        print("Reason:", reason.replace('\n', ' '))
        
        # Conclude
        input_text = processor.apply_chat_template(msgs(reason+ "\nTask: Determine if a patient is being turned by someone else. Your answer should be either 'Yes' or 'No'."), add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        res = model.generate(**inputs, max_new_tokens=max_new_tokens)
        res = processor.decode(res[0])
        res = "False" if "No" in res.split("<|end_header_id|>")[-1] else "True" # We want to have a higher Recall rate, so rather than finding Yes, we want to find No.
        print("Cut:", res)
        
        reason = reason.replace('\n', ' ')
        data.append([image_path, res, reason])
        result[image_path] = (res, reason)
        if show_img:
            display(HTML(f'<img src="{Path(input_path).joinpath(image_path) if os.path.isdir(input_path) else image_path }" style="width:30%;">'))
    data.sort()
    
    # if output_file is specified, it generates tsv file
    if output_file is not None:
        data_frame = pd.DataFrame(data, columns=['Image', 'llm_evaluation', 'Reason'])
        data_frame.to_csv(output_file, sep = '\t', index = exportedfile_indexing, encoding = 'utf-8')
    return result 

if __name__ == "__main__" :
    print(img2text("frames", output_file = "result.tsv", show_img = False))
