# Copyright (c) 2023 Jing Du (thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
#import wave
import librosa
from stream_kws_ctc import KeyWordSpotter

kws_xiaowen = KeyWordSpotter(ckpt_path='model/hixiaowen/avg_30.pt',
                     config_path='model/hixiaowen/config.yaml',
                     token_path='model/tokens.txt',
                     lexicon_path='model/lexicon.txt',
                     threshold=0.02,
                     min_frames=5,
                     max_frames=250,
                     interval_frames=50,
                     score_beam=3,
                     path_beam=20,
                     gpu=-1,
                     is_jit_model=False,)

kws_xiaowen.set_keywords("嗨小问,你好问问")

def detection(audio, kw):
    if kw=='hixiaowen' or kw=='nihaowenwen':
        kws=kws_xiaowen

    else:  # for other input data, we recommend xiaowen model
       kws=kws_xiaowen

    kws.reset_all()
    if audio is None:
        return "Input Error! Please enter one audio!"

    # with wave.open(audio, 'rb') as fin:
    #     assert fin.getnchannels() == 1
    #     wav = fin.readframes(fin.getnframes())

    y, _ = librosa.load(audio, sr=16000)
    # NOTE: model supports 16k sample_rate
    wav = (y * (1 << 15)).astype("int16").tobytes()

    # We inference every 0.3 seconds, in streaming fashion.
    interval = int(0.3 * 16000) * 2
    for i in range(0, len(wav), interval):
        chunk_wav = wav[i: min(i + interval, len(wav))]
        print("len(chunk_wav) = {}".format(len(chunk_wav)))
        result = kws.forward(chunk_wav)

        if 'state' in result and result['state']==1:
            keyword=result['keyword']
            start=result['start']
            end=result['end']
            txt = f'Activated: Detect {keyword} from {start} to {end} second.'
            return txt
    return "Deactivated."


# input
inputs = [
    gr.Audio(sources="microphone", type="filepath", label='Input audio'),
    gr.Radio(['hixiaowen', 'nihaowenwen', 'none'], label='kw')
]

output = gr.Textbox(label="Output Result")

examples = [
    ['examples/gongqu-4.5_0000.wav', 'none'],
    ['examples/neiwaizao-35.5h_0000.wav', 'none'],
    ['examples/neizao-4.5h_0000.wav', 'none'],
    ['examples/waizao-5.5h_0000.wav', 'none'],
    ['examples/0000c7286ebc7edef1c505b78d5ed1a3.wav', 'nihaowenwen'],
    ['examples/0000e12e2402775c2d506d77b6dbb411.wav', 'nihaowenwen'],
    ['examples/000af5671fdbaa3e55c5e2bd0bdf8cdd.wav', 'hixiaowen'],
    ['examples/000eae543947c70feb9401f82da03dcf.wav', 'hixiaowen'],
]

text = "Key Word Spotting | 关键词/唤醒词检测"

# description
description = (
    "KWS Demo! Support 'hixiaowen' and 'nihaowenwen'."
)

interface = gr.Interface(
    fn=detection,
    inputs=inputs,
    outputs=output,
    title=text,
    description=description,
    examples=examples,
    theme='huggingface',
)

interface.launch()