# Batch Inference Feature

This document describes the new batch inference capability added to IndexTTS.

## Overview

The batch inference feature allows you to process multiple audio generation tasks in a single operation, where each task consists of a triple of:
- **Reference audio** (speaker voice)
- **Emotion audio** (optional, for emotion control)  
- **Text** (content to synthesize)

## Features Added

### 1. Core Batch Inference Method

**File:** `indextts/infer_v2.py`

Added `infer_batch()` method to `IndexTTS2` class that:
- Accepts a list of batch input dictionaries
- Processes each item using the existing `infer()` method
- Provides progress tracking and error handling
- Returns list of output audio files or audio data

**Usage:**
```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(model_dir="checkpoints")

batch_inputs = [
    {
        'spk_audio_prompt': 'speaker1.wav',
        'text': 'Hello world!',
        'emo_audio_prompt': 'happy.wav',
        'emo_alpha': 1.0,
        'max_text_tokens_per_segment': 120
    },
    {
        'spk_audio_prompt': 'speaker2.wav', 
        'text': 'Another sentence.',
        'emo_audio_prompt': None,
        'emo_alpha': 1.0
    }
]

outputs = tts.infer_batch(batch_inputs, output_dir="batch_outputs")
```

### 2. Web UI Batch Processing

**File:** `webui.py`

Added new "批量生成" (Batch Generation) tab with:
- CSV input interface for batch data
- Output directory configuration (auto or custom)
- Progress tracking during batch processing
- Results summary and sample audio output

**CSV Format:**
```csv
spk_audio_prompt,text,emo_audio_prompt,emo_alpha,emo_text,max_text_tokens_per_segment
examples/speaker1.wav,"Hello, how are you?",examples/happy.wav,1.0,"cheerful",120
examples/speaker2.wav,"This is another test.",examples/sad.wav,0.8,"melancholic",100
```

**Required columns:**
- `spk_audio_prompt`: Path to speaker reference audio
- `text`: Text content to synthesize

**Optional columns:**
- `emo_audio_prompt`: Path to emotion reference audio
- `emo_alpha`: Emotion weight (0.0-1.0, default 1.0)
- `emo_text`: Emotion description text
- `max_text_tokens_per_segment`: Max tokens per segment (default 120)

## Benefits

1. **Efficiency**: Process multiple audio files in one operation
2. **Consistency**: Same generation parameters applied across all items
3. **Progress Tracking**: Monitor batch processing progress
4. **Error Handling**: Continue processing even if individual items fail
5. **Flexible Input**: Support various combinations of speaker/emotion/text

## Example Use Cases

1. **Dataset Generation**: Create training data with multiple speaker-text combinations
2. **Voice Testing**: Test different speakers with same text content
3. **Emotion Variations**: Generate same text with different emotional expressions
4. **Bulk Content**: Process multiple sentences/paragraphs efficiently

## Usage Tips

1. **File Paths**: Use relative paths from the IndexTTS directory or absolute paths
2. **Batch Size**: Start with smaller batches (5-10 items) to test setup
3. **Memory**: Large batches may require more GPU memory
4. **Error Handling**: Check the results summary for failed items
5. **Output Organization**: Use descriptive output directory names

## API Reference

### `IndexTTS2.infer_batch()`

**Parameters:**
- `batch_inputs` (list): List of dictionaries with batch item parameters
- `output_dir` (str, optional): Directory to save outputs
- `verbose` (bool): Enable verbose logging
- `**generation_kwargs`: Additional generation parameters

**Returns:**
- List of output paths (if output_dir specified) or audio data tuples

### Batch Item Dictionary

**Required keys:**
- `spk_audio_prompt` (str): Speaker audio file path
- `text` (str): Text to synthesize

**Optional keys:**
- `emo_audio_prompt` (str): Emotion audio file path
- `emo_alpha` (float): Emotion weight
- `emo_vector` (list): Emotion vector
- `use_emo_text` (bool): Use emotion text
- `emo_text` (str): Emotion description
- `use_random` (bool): Use random emotion
- `max_text_tokens_per_segment` (int): Max tokens per segment
- `interval_silence` (int): Silence interval between segments