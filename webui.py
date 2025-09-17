import json
import os
import sys
import threading
import time

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_BASE = EMO_CHOICES[:3]  # 基础选项
EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES  # 全部选项（包括文本描述）

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             example.get("emo_text") is not None]
                             )

def normalize_emo_vec(emo_vec):
    # emotion factors for better user experience
    k_vec = [0.75,0.70,0.80,0.80,0.75,0.75,0.55,0.45]
    tmp = np.array(k_vec) * np.array(emo_vec)
    if np.sum(tmp) > 0.8:
        tmp = tmp * 0.8/ np.sum(tmp)
    return tmp.tolist()

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        # normalize emo_alpha for better user experience
        emo_weight = emo_weight * 0.8
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = normalize_emo_vec(vec)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def gen_batch(batch_data, output_dir_option, custom_output_dir, progress=gr.Progress()):
    """
    Process batch inference for multiple triples of (reference, emotion, text).
    
    Args:
        batch_data (str): CSV/JSON string or file containing batch data
        output_dir_option (str): "auto" or "custom"
        custom_output_dir (str): Custom output directory path
        progress: Gradio progress tracker
    """
    import csv
    import io
    
    if not batch_data or not batch_data.strip():
        return "❌ No batch data provided", None
    
    # Set gradio progress
    tts.gr_progress = progress
    
    try:
        # Parse batch data (expecting CSV format with columns: spk_audio_prompt, text, emo_audio_prompt, emo_weight, etc.)
        batch_inputs = []
        
        # Try to parse as CSV
        try:
            csv_reader = csv.DictReader(io.StringIO(batch_data.strip()))
            for row in csv_reader:
                if 'spk_audio_prompt' in row and 'text' in row:
                    batch_item = {
                        'spk_audio_prompt': row.get('spk_audio_prompt', '').strip(),
                        'text': row.get('text', '').strip(),
                        'emo_audio_prompt': row.get('emo_audio_prompt', '').strip() or None,
                        'emo_alpha': float(row.get('emo_alpha', '1.0')),
                        'emo_text': row.get('emo_text', '').strip() or None,
                        'max_text_tokens_per_segment': int(row.get('max_text_tokens_per_segment', '120')),
                    }
                    if batch_item['spk_audio_prompt'] and batch_item['text']:
                        batch_inputs.append(batch_item)
        except Exception as e:
            return f"❌ Error parsing CSV data: {str(e)}", None
        
        if not batch_inputs:
            return "❌ No valid batch inputs found. Please ensure CSV has 'spk_audio_prompt' and 'text' columns.", None
        
        # Determine output directory
        if output_dir_option == "custom" and custom_output_dir.strip():
            output_dir = custom_output_dir.strip()
        else:
            output_dir = os.path.join("outputs", f"batch_{int(time.time())}")
        
        # Process batch
        outputs = tts.infer_batch(
            batch_inputs=batch_inputs,
            output_dir=output_dir,
            verbose=cmd_args.verbose,
            # Default generation parameters
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            length_penalty=0.0,
            num_beams=3,
            repetition_penalty=10.0,
            max_mel_tokens=1500
        )
        
        # Create result summary
        successful_outputs = [o for o in outputs if o is not None]
        failed_count = len(outputs) - len(successful_outputs)
        
        result_message = f"✅ Batch processing completed!\n"
        result_message += f"📊 Total items: {len(batch_inputs)}\n"
        result_message += f"✅ Successful: {len(successful_outputs)}\n"
        result_message += f"❌ Failed: {failed_count}\n"
        result_message += f"📁 Output directory: {output_dir}\n"
        
        if successful_outputs:
            result_message += f"\n📋 Generated files:\n"
            for i, output_path in enumerate(successful_outputs[:10]):  # Show first 10
                if output_path:
                    result_message += f"  {i+1}. {os.path.basename(output_path)}\n"
            if len(successful_outputs) > 10:
                result_message += f"  ... and {len(successful_outputs) - 10} more files\n"
        
        # Return the first successful output as sample
        sample_output = None
        for output in successful_outputs:
            if output and os.path.exists(output):
                sample_output = output
                break
                
        return result_message, sample_output
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during batch processing: {str(e)}\n"
        if cmd_args.verbose:
            error_msg += f"\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio")
        experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"),value=False)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_BASE,
                    type="index",
                    value=EMO_CHOICES_BASE[0],label=i18n("情感控制方式"))
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))


        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.8, step=0.01)

        with gr.Accordion(i18n("高级生成参数设置"), open=False,visible=False) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]
        
        if len(example_cases) > 2:
            example_table = gr.Examples(
                examples=example_cases[:-2],
                examples_per_page=20,
                inputs=[prompt_audio,
                        emo_control_method,
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,experimental_checkbox]
            )
        elif len(example_cases) > 0:
            example_table = gr.Examples(
                examples=example_cases,
                examples_per_page=20,
                inputs=[prompt_audio,
                        emo_control_method,
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, experimental_checkbox]
            )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_select(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def on_experimental_change(is_exp):
        # 切换情感控制选项
        # 第三个返回值实际没有起作用
        if is_exp:
            return gr.update(choices=EMO_CHOICES_EXPERIMENTAL, value=EMO_CHOICES_EXPERIMENTAL[0]), gr.update(visible=True),gr.update(value=example_cases)
        else:
            return gr.update(choices=EMO_CHOICES_BASE, value=EMO_CHOICES_BASE[0]), gr.update(visible=False),gr.update(value=example_cases[:-2])

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox],
        outputs=[emo_control_method, advanced_settings_group,example_table.dataset]  # 高级参数Accordion
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    # Batch Processing Tab
    with gr.Tab(i18n("批量生成")):
        gr.Markdown(f"### {i18n('批量音频生成')}")
        gr.Markdown(i18n("上传包含多个音频生成任务的CSV文件，支持批量处理多组 (参考音频, 情感音频, 文本) 的组合。"))
        
        with gr.Row():
            with gr.Column(scale=2):
                batch_input = gr.TextArea(
                    label=i18n("批量数据 (CSV格式)"),
                    placeholder="""spk_audio_prompt,text,emo_audio_prompt,emo_alpha,emo_text,max_text_tokens_per_segment
examples/sample_prompt.wav,"Hello world, this is a test.",examples/emo_sample.wav,1.0,"happy",120
examples/sample_prompt2.wav,"Another test sentence.",examples/emo_sample2.wav,0.8,"excited",100""",
                    lines=10,
                    info=i18n("请输入CSV格式的批量数据。必需列: spk_audio_prompt, text。可选列: emo_audio_prompt, emo_alpha, emo_text, max_text_tokens_per_segment")
                )
                
                with gr.Row():
                    output_dir_option = gr.Radio(
                        choices=["auto", "custom"],
                        value="auto",
                        label=i18n("输出目录选择"),
                        info=i18n("auto: 自动生成目录，custom: 自定义目录")
                    )
                    custom_output_dir = gr.Textbox(
                        label=i18n("自定义输出目录"),
                        placeholder="outputs/my_batch",
                        visible=False
                    )
                
                batch_gen_button = gr.Button(i18n("开始批量生成"), variant="primary")
                
            with gr.Column(scale=1):
                batch_result = gr.TextArea(
                    label=i18n("批量处理结果"),
                    lines=10,
                    interactive=False
                )
                batch_sample_output = gr.Audio(
                    label=i18n("示例输出"),
                    visible=True
                )
        
        # Add example CSV data
        with gr.Accordion(i18n("CSV格式说明"), open=False):
            gr.Markdown("""
            **必需列:**
            - `spk_audio_prompt`: 音色参考音频文件路径
            - `text`: 要合成的文本内容
            
            **可选列:**
            - `emo_audio_prompt`: 情感参考音频文件路径
            - `emo_alpha`: 情感权重 (0.0-1.0, 默认1.0)
            - `emo_text`: 情感描述文本
            - `max_text_tokens_per_segment`: 分句最大Token数 (默认120)
            
            **示例CSV:**
            ```csv
            spk_audio_prompt,text,emo_audio_prompt,emo_alpha,emo_text,max_text_tokens_per_segment
            examples/speaker1.wav,"Hello, how are you?",examples/happy.wav,1.0,"cheerful",120
            examples/speaker2.wav,"This is another test.",examples/sad.wav,0.8,"melancholic",100
            ```
            """)
        
        # Event handlers for batch tab
        def update_custom_dir_visibility(option):
            return gr.update(visible=(option == "custom"))
        
        output_dir_option.change(
            update_custom_dir_visibility,
            inputs=[output_dir_option],
            outputs=[custom_output_dir]
        )
        
        batch_gen_button.click(
            gen_batch,
            inputs=[batch_input, output_dir_option, custom_output_dir],
            outputs=[batch_result, batch_sample_output]
        )


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True)
