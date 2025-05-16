import argparse
import torch
import os
from io import BytesIO
from pydub import AudioSegment
import librosa
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import logging
import json
from pathlib import Path
from loguru import logger

torch.backends.cuda.enable_flash_sdp(True)
EXPECTED_SAMPLING_RATE = 16000


def load_model(model_path: str):
    """
    Load the model and processor, explicitly setting sampling_rate.
    """
    logger.info(f"Loading model from: {model_path}")
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        model = torch.compile(model)
        processor = AutoProcessor.from_pretrained(model_path, sampling_rate=EXPECTED_SAMPLING_RATE)

        if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
            loaded_sr = processor.feature_extractor.sampling_rate
            logger.info(
                f"Processor loaded. Expected SR: {EXPECTED_SAMPLING_RATE}, Actual SR in feature_extractor: {loaded_sr}"
            )
            if loaded_sr != EXPECTED_SAMPLING_RATE:
                logger.warning(
                    f"Processor's feature_extractor.sampling_rate ({loaded_sr}) differs from EXPECTED_SAMPLING_RATE ({EXPECTED_SAMPLING_RATE})."
                )
        else:
            logger.warning("Could not verify processor's feature_extractor.sampling_rate.")

        logger.info("Model and processor loaded successfully!")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model or processor from {model_path}: {e}", exc_info=True)
        return None, None


def compute_hls_score(model, processor, audio_url: str):
    """
    Compute the HLS (Human-Likeness Score) for a given audio file.
    """
    if not (audio_url.startswith("http://") or audio_url.startswith("https://")):
        assert os.path.exists(audio_url), f"Audio file not found at: {audio_url}"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_url},
                {
                    "type": "text",
                    "text": "Suppose you are an expert in judging and evaluating the quality of audios.\nYour judgement should be chosen from the following three catefories: Good, Fair, Bad.\nBad: The sound clearly has synthetic or machine characteristics. There is significant distortion, monotonous intonation, or unnatural speech speed. The pronunciation and intonation lack variation, and the listening experience is more mechanized.\nFair: The sound has some human qualities but still does not seem natural enough. The intonation and speech speed may be incoherent or appear rigid. There is slight distortion or artifacts that make it difficult to determine if it is machine-generated.\nGood: The sound is completely natural and almost indistinguishable from a real person. The intonation, speech speed, and emotions are smooth and authentic. No obvious artifacts or distortions are present.\nPlease note, only the category (Good, Fair, Bad) needs to be output.\nNow please rate this audio:",
                },
            ],
        },
    ]
    try:
        if audio_url.endswith(".mp3"):
            if audio_url.startswith("http://") or audio_url.startswith("https://"):
                response = urlopen(audio_url)
                audio_content = BytesIO(response.read())
                audio = AudioSegment.from_mp3(audio_content)
            else:
                audio = AudioSegment.from_mp3(audio_url)
            wav_data = BytesIO()
            audio.export(wav_data, format="wav")
            wav_data.seek(0)
        elif audio_url.endswith(".wav") or ".wav" in audio_url:
            if os.path.exists(audio_url):
                with open(audio_url, "rb") as f:
                    wav_data = BytesIO(f.read())
            else:
                wav_data = BytesIO(urlopen(audio_url).read())
        else:
            logger.error(f"Unsupported audio format: {audio_url}. Please provide a .mp3 or .wav file.")
            return None

        audios, sr = librosa.load(wav_data, sr=EXPECTED_SAMPLING_RATE)
        if sr != EXPECTED_SAMPLING_RATE:
            logger.warning(
                f"Librosa loaded audio with SR {sr}, but expected {EXPECTED_SAMPLING_RATE}. Resampling occurred."
            )

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, audio=audios, sampling_rate=EXPECTED_SAMPLING_RATE, return_tensors="pt", padding=True
        )
        inputs = inputs.to(model.device)

        anchor_token_ids = [15216, 60795, 17082]

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_logits=True,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        logits = outputs.logits[0]
        logits_gathered = torch.gather(logits[0], dim=-1, index=torch.tensor(anchor_token_ids).to(logits.device))
        probs_gathered = torch.softmax(logits_gathered, dim=-1)

        weights = torch.tensor([3.0, 2.0, 1.0], device=logits.device)
        weighted_mos = torch.sum(probs_gathered * weights, dim=0).item()

        return weighted_mos
    except Exception as e:
        logger.error(f"Error processing audio {audio_url}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute HLS score for an audio file or a directory of audio files.")
    parser.add_argument("--audio_path", type=str, required=False, help="Path to a single audio file (.mp3 or .wav).")
    parser.add_argument("--audio_dir", type=str, required=False, help="Path to a directory containing audio files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Qwen2Audio model.")
    parser.add_argument(
        "--output_file", type=str, default="hls_scores.jsonl", help="Path to save the output JSONL file."
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recompute scores for all files, even if already processed.",
    )
    args = parser.parse_args()

    if not args.audio_path and not args.audio_dir:
        logger.error("Either --audio_path or --audio_dir must be provided.")
        exit(1)

    if args.audio_path and args.audio_dir:
        logger.error("Both --audio_path and --audio_dir provided. Please provide only one.")
        exit(1)

    model, processor = load_model(args.model_path)
    if model is None or processor is None:
        logger.error("Failed to load model or processor. Exiting.")
        exit(1)

    processed_ids = set()
    if not args.force_recompute and os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if "id" in item:
                            processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading existing output file {args.output_file}: {e}", exc_info=True)
            exit(1)
    elif args.force_recompute and os.path.exists(args.output_file):
        logger.info("Force recompute mode enabled. Ignoring previously processed files.")

    # Open the output file in append mode
    try:
        output_f = open(args.output_file, "a", encoding="utf-8")
    except IOError as e:
        logger.error(f"Failed to open output file {args.output_file} for appending: {e}", exc_info=True)
        exit(1)

    logger.warning(f"Already process {len(processed_ids)} files")

    processed_count = 0
    skipped_count = 0

    if args.audio_path:
        audio_id = os.path.basename(args.audio_path)
        if not args.force_recompute and audio_id in processed_ids or "ziyan" in audio_id:
            logger.info(f"Skipping already processed audio: {args.audio_path}")
            skipped_count += 1
        else:
            logger.info(f"Evaluating audio: {args.audio_path}")
            score = compute_hls_score(model, processor, args.audio_path)
            if score is not None:
                result_item = {"id": audio_id, "hls_score": round(score, 3)}
                output_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                output_f.flush()  # Ensure it's written immediately
                logger.info(f"Computed HLS Score for {args.audio_path}: {score:.3f}")
                processed_count += 1
            else:
                logger.error(f"Failed to compute HLS score for {args.audio_path}")

    elif args.audio_dir:
        logger.info(f"Evaluating audio files in directory: {args.audio_dir}")
        if not os.path.isdir(args.audio_dir):
            logger.error(f"Provided audio_dir is not a valid directory: {args.audio_dir}")
            output_f.close()
            exit(1)

        audio_files = []
        for root, _, files in os.walk(args.audio_dir):
            for file_name in files:
                if file_name.lower().endswith((".mp3", ".wav")):
                    audio_files.append(os.path.join(root, file_name))

        if not audio_files:
            logger.info(f"No .mp3 or .wav files found in {args.audio_dir}")
        else:
            logger.info(f"Found {len(audio_files)} audio files in directory.")
            to_process = []
            for audio_file_path in audio_files:
                audio_id = os.path.basename(audio_file_path)
                if (not args.force_recompute and audio_id in processed_ids) or "ziyan" in audio_id:
                    # logger.debug(f"Skipping already processed audio: {audio_file_path}")
                    skipped_count += 1
                else:
                    to_process.append(audio_file_path)

            logger.info(f"Skipping {skipped_count} already processed files. Processing {len(to_process)} new files.")
            for audio_file_path in tqdm(to_process, desc="Processing audio files"):
                # logger.info(f"Evaluating audio: {audio_file_path}")
                score = compute_hls_score(model, processor, audio_file_path)
                if score is not None:
                    audio_id = os.path.basename(audio_file_path)
                    result_item = {"id": audio_id, "hls_score": round(score, 3)}
                    output_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                    output_f.flush()  # Ensure it's written immediately
                    # logger.info(f"Computed HLS Score for {audio_file_path}: {score:.3f}")
                    processed_count += 1
                else:
                    logger.error(f"Failed to compute HLS score for {audio_file_path}")

    output_f.close()

    if processed_count > 0:
        logger.info(f"{processed_count} result(s) saved to {args.output_file}")
    else:
        logger.info("No results were processed and saved.")

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} already processed audio files.")

    logger.info(f"Processing finished. Processed {processed_count} files, skipped {skipped_count} files.")
