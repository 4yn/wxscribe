#!/usr/bin/env python

import argparse
from collections import namedtuple
import os
import sys

DESCRIPTION = """\
Transcribe any audio file and convert to a Microsoft Word document.
Wrapper around ffmpeg, whisperx (whisper + pyannote) and pandoc.
Assembled by @4yn.

Requires whisperx, ffmpeg-python and pandoc pip packages + pandoc=2.x installed on system.
`pip install git+https://github.com/m-bain/whisperx.git ffmpeg-python pandoc`
"""

def main():
    parser = argparse.ArgumentParser(
        prog='wxscribe',
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('audio_files', nargs="+")
    parser.add_argument('--hf-token', dest="hf_token", help="Huggingface token for pyannote model. Defaults to `HUGGING_FACE_HUB_TOKEN` environment variable.", default=None)
    parser.add_argument('--model-size', '-m', default="medium", choices=["tiny", "base", "small", "medium", "large", "large-v2"], help="Whisper model size to use. Larger models use GPU memory and may give better quality transcription. Decrease if out of memory or if the transcription is 'too good' e.g. fixes grammatical errors that should be preserved.")
    parser.add_argument('--batch-size', '-b', type=int, default=16, choices=[1, 2, 4, 8, 16, 32], help="Whisper model batch size to use. Larger batch sizes use more GPU memory and will process text faster. Decrease if out of memory.")
    parser.add_argument('--language', '-l', default="en", help="Language to transcribe. Full list of languages is available at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py .")
    parser.add_argument('--show-timestamps', '-t', action="store_true", dest="show_timestamps", help="Add a timestamp at the start of each paragraph")
    parser.add_argument('--separate-paragraphs', action="store_true", dest="separate_paragraphs", help="Label the speaker's name for each paragraph even if they are speaking back-to-back")
    parser.add_argument('--no-speaker', action="store_true", dest="no_speaker", help="Do not add speaker names")
    args = parser.parse_args()

    if args.hf_token is None:
        args.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    dependency_check(args)

    for audio_file in args.audio_files:

        if not os.path.exists(audio_file):
            print(f"Audio file {audio_file} not found, stopping.")
            sys.exit(1)

        if audio_file[-4:] != ".mp3":
            audio_file = generate_mp3(audio_file)

        output_file = audio_file.rpartition(".")[0] + ".docx"    
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping")
            continue

        print(f"Running recognition on {audio_file}")
        recognition_result = recognition_with_diarize_pipeline(audio_file, model_size=args.model_size, batch_size=args.batch_size, language=args.language, hf_token=args.hf_token)
        tw = TranscriptWriter()
        tw.parse_result(recognition_result)
        print(f"Saving to {output_file}")
        tw.format_transcript(output_file, include_timestamps=args.show_timestamps, merge_speaker=not args.separate_paragraphs, no_speaker=args.no_speaker)

def dependency_check(args):
    import subprocess

    try:
        subprocess.check_output(["ffmpeg", "-version"])
    except Exception:
        print("ffmpeg not found, please install ffmpeg.")
        sys.exit(1)

    try:
        res = subprocess.check_output(["pandoc", "-v"])
        if b"pandoc 2." not in res:
            pandoc_ver = res.partition(b'\n')[0].decode()
            print(f"Found {pandoc_ver} instead of pandoc 2.x, please install pandoc 2.x.")
            sys.exit(1)
    except Exception:
        print("pandoc not found, please install pandoc 2.x.")
        sys.exit(1)

    try:
        import ffmpeg
    except ImportError:
        print("Python package ffmpeg not found, install with `pip install ffmpeg-python`")
        sys.exit(1)

    try:
        import pandoc
    except ImportError:
        print("Python package pandoc not found, install with `pip install pandoc`")
        sys.exit(1)
    
    if args.hf_token is None:
        print("No huggingface token provided, please create an account and accept the ToC of pyannote at https://huggingface.co/pyannote/speaker-diarization, then generate an acccount token at https://huggingface.co/settings/tokens.")
        sys.exit(1)

    try:
        import whisperx
    except ImportError:
        print("Python package whisperx not found, install with `pip install git+https://github.com/m-bain/whisperx.git`")
        sys.exit(1)

def generate_mp3(audio_file):
    import ffmpeg
    audio_file_mp3 = audio_file.rpartition(".")[0] + ".mp3"
    if os.path.exists(audio_file_mp3):
        print(f"Using mp3 {audio_file_mp3}")
        return audio_file_mp3
    print(f"Converting {audio_file} to {audio_file_mp3}")
    (
        ffmpeg
            .input(audio_file)
            .output(audio_file_mp3, audio_bitrate="128k", acodec="libmp3lame")
            .run(quiet=True)
    )
    return audio_file_mp3

def recognition_with_diarize_pipeline(audio_file, batch_size=16, compute_type="float16", model_size="medium", device="cuda", language="en", hf_token=None):
    # hf_tokne = "hf_insert_your_api_key_here________" # if you want to hardcode it
    if hf_token is None:
        raise ValueError("Please create a huggingface account, agree to the pyannote T&Cs and provide your token to transcribe")
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    import logging
    logging.getLogger("lightning").setLevel(logging.ERROR)

    import whisperx
    import gc
    import torch

    model_whisper = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(audio_file)
    result = model_whisper.transcribe(audio, batch_size=batch_size)

    gc.collect()
    torch.cuda.empty_cache()
    del model_whisper

    model_align, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_align, metadata, audio, device, return_char_alignments=False)

    gc.collect()
    torch.cuda.empty_cache()
    del model_align

    model_diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = model_diarize(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    gc.collect()
    torch.cuda.empty_cache()
    del model_diarize

    return result

Paragraph = namedtuple("Paragraph", "speaker time text")

class TranscriptWriter:
    def __init__(self):
        pass

    def parse_result(self, result, seconds_for_paragraph=1):
        transcript = []

        current_speaker = None
        current_text = []
        current_start_time = -999
        current_end_time = -999

        for segment in result["segments"]:
            seconds_since_last_segment = segment["start"] - current_end_time
            speaker, text = segment.get("speaker", "SPEAKER_UNKNOWN"), segment["text"].strip()

            if (
                (speaker != current_speaker or seconds_since_last_segment > seconds_for_paragraph) and
                current_speaker is not None
            ):
                transcript.append(Paragraph(
                    current_speaker,
                    current_start_time,
                    " ".join(current_text)
                ))
                current_text = []
            if speaker != current_speaker:
                current_start_time = segment["start"]
                current_speaker = speaker
            current_text.append(text.strip())
            current_end_time = segment["end"]
        transcript.append(Paragraph(
            current_speaker,
            current_start_time,
            " ".join(current_text)
        ))
        self.transcript = transcript

    def format_timestamp(self, t):
        t = round(t)
        s, t = t % 60, t // 60
        m, h = t % 60, t // 60
        return f"{h:02}:{m:02}:{s:02}"

    def format_speaker(self, s):
        if s == "SPEAKER_UNKNOWN":
            return "(Unknown Speaker)"
        return f"(Speaker {int(s.partition('_')[2]) + 1})"

    def format_transcript(self, filename=None, fix_caps=True, include_timestamps=False, merge_speaker=True, no_speaker=False):
        transcript = self.transcript
        lines = []

        current_speaker = None
        for speaker, time, text in transcript:
            if (speaker != current_speaker or not merge_speaker) and not no_speaker:
                lines.append(f"{self.format_speaker(speaker)}:")
            if include_timestamps:
                lines.append(self.format_timestamp(time))
            # first capitalize
            if fix_caps:
                text = text[:1].upper() + text[1:]

            lines.append(f"{text}")

            current_speaker = speaker
        
        formatted_transcript = "\n\n".join(lines)

        if filename is not None:
            import pandoc

            if filename[-5:] != ".docx":
                raise ValueError("Filename should end in .docx")
            doc = pandoc.read(formatted_transcript)
            pandoc.write(doc, filename, format="docx")

        return formatted_transcript
    
if __name__ == "__main__":
    main()