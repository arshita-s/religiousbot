import json, os, sys
import gpt_2_simple as gpt
import numpy as np
#import tweepy
from typing import Optional, List

"""
def generate_tweet(checkpoint_dir: str,
    length: int,
    temperature: float,
    destination_path: Optional[str],
    prefix: Optional[str],
    return_as_list: bool = False,) -> List[str]:
    sess = gpt.start_tf_sess()
    gpt.load_gpt2(sess, checkpoint_dir=checkpoint_dir)
    text = gpt.generate(
        sess,
        checkpoint_dir=checkpoint_dir,
        length=length,
        temperature=temperature,
        destination_path=destination_path,
        prefix=prefix,
        return_as_list=return_as_list,
    )
    return text
"""

model_name = "124M"
checkpoint_dir = "checkpoint"
destination_path = None
temperature = 1
prefix = ''
return_as_list = False
length = 30

if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt.download_gpt2(model_name=model_name)

sess = gpt.start_tf_sess()

num_steps = 30

text_path = "text.txt"

gpt.finetune(sess, text_path, model_name=model_name, steps=num_steps)

gpt.generate(sess)

gpt.load_gpt2(sess, checkpoint_dir=checkpoint_dir)
text = gpt.generate(
    sess,
    checkpoint_dir=checkpoint_dir,
    length=length,
    temperature=temperature,
    destination_path=destination_path,
    prefix=prefix,
    return_as_list=True
)
