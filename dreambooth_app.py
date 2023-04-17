import os
from dataclasses import dataclass
from pathlib import Path
from fastapi import FastAPI
import modal

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"
stub = modal.Stub(name="example-dreambooth-app")

GIT_SHA = "ed616bd8a8740927770eebe017aedb6204c6105f"
image = (
    modal.Image.debian_slim(python_version="3.10").pip_install(
        "accelerate",
        "datasets",
        "ftfy",
        "gradio~=3.10",
        "smart_open",
        "transformers",
        "torch",
        "torchvision",
        "triton"
    ).pip_install("xformers", pre=True).apt_install(
        "git"
    ).run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)

volume = modal.SharedVolume().persist("dreambooth-finetuning-vol")
MODEL_DIR = Path("/model")


@dataclass
class SharedConfig:
    """Configuration information shared across project components"""
    instance_name: str = "Krzysztof Szafranek"
    class_name: str = "person"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the fine-tuning step"""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of "
    postfix: str = ""

    # locator of for plaintext file with URLs for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # identifier for pretrained model on Hugging Face
    model_name: str = "stabilityai/stable-diffusion-2-1"

    resolution: int = 768
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 600
    checkpointing_steps: int = 1000


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for reference"""

    num_interference_steps: int = 50
    guidance_scale: float = 7.5


IMG_PATH = Path("/img")


def load_images(image_urls):
    import PIL.Image
    from smart_open import open

    os.makedirs(IMG_PATH, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            img = PIL.Image.open(f)
            img.save(IMG_PATH / f"{ii}.jpg")
    print("Images loaded")
    return IMG_PATH


@stub.function(
    image = image,
    gpu="A100",
    shared_volumes={
        str(
            MODEL_DIR
        ): volume
    },
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("huggingface")],
)

def train(instance_example_urls, config=TrainConfig()):
    import subprocess
    import huggingface_hub
    from accelerate.utils import write_basic_config
    from transformers import CLIPTokenizer

    # set up runner-local image and shared model weights
    img_path = load_images(instance_example_urls)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # set up Hugging Face accelerate library for fast training
    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    try:
        CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    except OSError as e:
        license_error_msg = f"Unable to load tokenizer. Access to this model requires acceptance of the license on Hugging Face here: https://huggingface.co/{config.model_name}."
        raise Exception(license_error_msg) from e

    # define the training prompt
    instance_phrase = f"{config.instance_name} {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    # run training – see Hugging Face accelerate docs for details
    subprocess.run(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth.py",
            "--train_text_encoder",
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
        ],
        check=True
    )


class Model:
    def __enter__(self):
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        # set up a Hugging Face inference pipeline using our model
        ddim = DDIMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            scheduler=ddim,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        self.pipe = pipe

    @stub.function(
        image=image,
        gpu="A100",
        shared_volumes={str(MODEL_DIR): volume},
    )
    def inference(self, text, config: AppConfig):
        image = self.pipe(
            text,
            num_inference_steps=config.num_interference_steps,
            guidance_scale=config.guidance_scale
        ).images[0]
        return image


@stub.function(
    image=image,
    concurrency_limit=3,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@stub.asgi_app()
def fastapi_app(config=AppConfig()):
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call to the GPU inference function on Modal
    def go(text):
        return Model().inference.call(text, config)

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"A painting of {instance_phrase.title()} With a Pearl Earring, by Vermeer,"
        f"Oil painting of {instance_phrase} flying through space as an astronaut",
        f"A painting of {instance_phrase} in cyberpunk city. Character design by Cory Loftis. Volumetric lighting."
        f"Drawing of {instance_phrase} high quality, cartoon, path traced, by Studio Ghibli and Don Bluth",
    ]

    model_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{model_docs_url}/ex/dreambooth_app"

    description = f"""Describe what they're doing or how a particular artist or style would depict the subject.
Be fantastical! Try the examples below for inspiration.

### Learn how to make your own [here]({modal_example_url}).    
    """

    # Add a Gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title=f"Generate images of {instance_phrase}.",
        description=description,
        examples=example_prompts,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/"
    )


@stub.local_entrypoint()
def run():
    with open(TrainConfig().instance_example_urls_file) as f:
        instance_example_urls = [line.strip() for line in f.readlines()]
        train.call(instance_example_urls)
