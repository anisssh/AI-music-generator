import base64
from typing import List
import uuid
import modal
import os
from pydantic import BaseModel
import requests
import torch
import boto3
from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT

app = modal.App("music-generator")

image = ( 
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
  .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", "cd /tmp/ACE-Step && pip install ."])
          .env({"HF_HOME" : "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-gen-secret")

class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str

class GenerateWithCustomLyrics(AudioGenerationBase):
    prompt: str
    lyrics: str

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str

class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]

class GenerateMusicResponse(BaseModel):
    audio_data: str

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15

)

class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from diffusers import AutoPipelineForText2Image
        #Music generation model
        self.music_model = ACEStepPipeline(
            checkpoint_dir = "/models",
            dtype = "bfloat16",
            torch_compile= False,
            cpu_offload= False,
            overlapped_decode= False
        )
        #Large Langauge model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface" 
        )
        #Stable diffusion models
        self.image_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str):
        messages = [
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_prompt(self, description: str):
        #insert description into template
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt = description)

        # run LLM inference and return that
        return self.prompt_qwen(full_prompt)

    def generate_prompt(self, description: str):
        #insert description into template
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description = description)    

        # run LLM inference and return that
        return self.prompt_qwen(full_prompt)
    
    def generate_categories(self, description: str) -> List[str]:
        prompt = f"Based on the following music description, List 3-4 relevant genres or categories as comma seperated list. For example: Pop, Hip-hop, Rock, 90s. Description: '{description}'"

        response_text = self.prompt_qwen(prompt)
        categories = [cat.strip() 
                      for cat in response_text.split(",") if cat.strip()]
        return categories


    def generate_and_upload_to_s3(
            self,
            prompt: str,
            lyrics: str,
            instrumental: bool,
            audio_duration: float,
            infer_step: int,
            guidance_scale: float,
            description_for_categories: str,
            seed: int
    ) -> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"prompt: \n{prompt}")

        #AWS
        #Create S3 bucket, store thumbnail, songs
        #IAM users
        #Backend (modal): PutObject
        #Frontend- Nextjs: GetObject, ListObject

        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir,exist_ok = True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")
       
        self.music_model(
            prompt = prompt,
            lyrics= final_lyrics,
            audio_duration = audio_duration,
            infer_step = infer_step,
            guidance_scale = guidance_scale,
            save_path = output_path,
            manual_seeds = str(seed)

          )
        
        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)
    
    #thumbnail generation
        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(prompt=thumbnail_prompt, num_inference_steps =2 , guidance_scale =0.0,).images[0]

        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

    #Category generation 

        categories = self.generate_categories(description_for_categories)
        return GenerateMusicResponseS3(
            s3_key=audio_s3_key,
            cover_image_s3_key=image_s3_key,
            categories=categories
        )



    @modal.fastapi_endpoint(method="POST",requires_proxy_auth=True)
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir,exist_ok = True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")
       
        self.music_model(
            prompt = "r&b, soul, funk/soul",
            lyrics = "[verse]\nDancing through electric fires\nHeart is buzzing like live wires\nIn your arms I find desire\nFeel the beat as we get higher\n\n[chorus]\nElectric love in the night sky\nWe’re gonna soar baby you and I\nDrop the bass let the rhythm fly\nFeel the heat and don't ask why\n\n[verse]\nWhisper secrets that make me blush\nUnder the neon city hush\nYour touch gives me such a rush\nTurn it up we're feeling lush\n\n[chorus]\nElectric love in the night sky\nWe’re gonna soar baby you and I\nDrop the bass let the rhythm fly\nFeel the heat and don't ask why\n\n[bridge]\nThrough the lights and the smoky haze\nI see you in a thousand ways\nLove's a script and we’re the play\nTurn the page stay till we sway\n\n[chorus]\nElectric love in the night sky\nWe’re gonna soar baby you and I\nDrop the bass let the rhythm fly\nFeel the heat and don't ask why",     
            audio_duration = 180,
            infer_step = 60,
            guidance_scale = 15,
            save_path = output_path

          )
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_b64)
    
    @modal.fastapi_endpoint(method="POST",requires_proxy_auth=True)
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:
        #Generating a prompt 
        prompt = self.generate_prompt(request.full_described_song)
        
        #Generating Lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_prompt(request.full_described_song)
        return self.generate_and_upload_to_s3(prompt=prompt,lyrics=lyrics,
                                               description_for_categories=request.full_described_song, **request.model_dump(exclude={"full_described_song"}))


        pass
    @modal.fastapi_endpoint(method="POST",requires_proxy_auth=True)
    def generate_from_lyrics(self,request: GenerateWithCustomLyrics) -> GenerateMusicResponseS3:
        return self.generate_and_upload_to_s3(prompt=request.prompt,lyrics=request.lyrics,
                                               description_for_categories=request.prompt, **request.model_dump(exclude={"prompt","lyrics"}))



        pass
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_described_lyrics(self,request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_prompt(request.full_described_song)
        return self.generate_and_upload_to_s3(prompt=request.prompt,lyrics=lyrics,
                                               description_for_categories=request.prompt, **request.model_dump(exclude={"described_lyrics","prompt"}))

        pass

headers = {
    "Modal-Key": "wk-NCqlVuyETsyIjSF2Vk8Uq9",
    "Modal-Secret":"ws-ciZdtfXC3a6BjAt1we7xER"
}

@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate_from_description.get_web_url()

    request_data = GenerateFromDescriptionRequest(
        full_described_song="Acoustic Ballad",
        guidance_scale=7.5
          )
    payload = request_data.model_dump() 

    response = requests.post(endpoint_url, json=payload)
    response.raise_for_status()
    result = GenerateMusicResponseS3(**response.json())

    print(
        f"Success: {result.s3_key}, {result.cover_image_s3_key} {result.categories}")

    #audio_bytes = base64.b64decode(result.audio_data)
    #output_filename = "generated.wav"
    #with open(output_filename,"wb") as f:
     #   f.write(audio_bytes)



