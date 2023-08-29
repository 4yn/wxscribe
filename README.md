# wxscribe

Transcribe any audio file and convert to a Microsoft Word document. Simple
wrapper around ffmpeg, [whisperx](https://github.com/m-bain/whisperx) (whisper +
pyannote) and pandoc, initially intended for interview transcription. Assembled
by @4yn.

[Quickstart colab notebook (make sure to pick the GPU runtime and to shutdown
once
done)](https://colab.research.google.com/drive/1wrAGHHXknA1GEOtvOltTYkBa_lsbPEDn?usp=sharing)

## Usage

```
./wxscribe.py audio_file_1 audio_file_2 audio_file_3 ...
```

You will need to [accept the pyannote model
ToC](https://huggingface.co/pyannote/speaker-diarization) and [generate a
huggingface model hub token](https://huggingface.co/settings/tokens) for the
script to work. Provide the token using command line option `--hf-token
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` or using the `HUGGING_FACE_HUB_TOKEN`
environment variable.

An audio file `my_audio.mp3` will be transcribed to `my_audio.docx`. If some
`my_audio.docx` already exists it will be skipped; delete the old document if
you want to repeat the transcription.

## Dependencies

```
pip install git+https://github.com/m-bain/whisperx.git ffmpeg-python pandoc
```

Also requires `pandoc=2.x` and `ffmpeg` installed on system. Alternatively, [use
this script on Google
Colab](https://colab.research.google.com/drive/1wrAGHHXknA1GEOtvOltTYkBa_lsbPEDn?usp=sharing)
as it already has these dependencies installed.

### Example Usage

```bash
$ yt-dlp https://www.youtube.com/watch?v=FAPeFmWmYQo -x -o "audio.%(ext)s"

[youtube] Extracting URL: https://www.youtube.com/watch?v=FAPeFmWmYQo
[youtube] FAPeFmWmYQo: Downloading webpage
[youtube] FAPeFmWmYQo: Downloading ios player API JSON
[youtube] FAPeFmWmYQo: Downloading android player API JSON
[youtube] FAPeFmWmYQo: Downloading m3u8 information
[info] FAPeFmWmYQo: Downloading 1 format(s): 251
[download] Destination: audio.webm
[download] 100% of    8.73MiB in 00:00:00 at 11.48MiB/s
[ExtractAudio] Destination: audio.opus
Deleting original file audio.webm (pass -k to keep)

$ wxscribe.py audio.opus --hf-token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Converting audio.opus to audio.mp3
Running recognition on audio.mp3
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.0.7. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint --file ../../.cache/torch/whisperx-vad-segmentation.bin`
Model was trained with pyannote.audio 0.0.1, yours is 2.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.0.1. Bad things might happen unless you revert torch to 1.x.
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.0.7. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint --file ../../.cache/torch/pyannote/models--pyannote--segmentation/snapshots/c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 2.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.0.1. Bad things might happen unless you revert torch to 1.x.
Saving to audio.docx
```

![sample transcription](./img/sample_transcription.png)

### Options

- `--model-size` / `--batch-size` / `--language`: Options for the whisper model.
  Adjust to change how much GPU memory is used and what language the transcribed
  text should be in.
- `--show-timestamps`: Print timestamps before each paragraph
- `--seperate-paragraphs`: Show speaker names before all paragraphs even if the
  same speaker continues speaking.
- `--no-speaker`: Do not show speaker names