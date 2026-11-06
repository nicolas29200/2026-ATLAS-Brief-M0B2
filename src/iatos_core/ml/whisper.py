import time
from typing import Any, Dict

import numpy as np
import soundfile as sf
from transformers import pipeline

from iatos_core.logger import logger


class WhisperProcessing:
    "Dictée vocale et transcription automatique de notes d'audit à chaud avec openai/whisper-tiny."

    def __init__(self):
        self.asr = pipeline('automatic-speech-recognition', model='openai/whisper-tiny')
        logger.info("Transcription automatique de notes d'audit initialise.")

    def transcription(self, audio_file) -> Dict[str, Any]:
        """Transcrit en texte le contenu d'un fichier audio.

        Args:
            audio_file: Chemin du fichier audio.

        Returns:
            Dict[str, Any]: Transcription, langue détectée et temps d'exécution.
        """
        start = time.perf_counter()

        # Decodage manuel (soundfile/libsndfile) pour eviter la dependance a ffmpeg
        audio_array, sampling_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        audio_array = audio_array.mean(axis=1).astype(np.float32)  # downmix vers mono

        result = self.asr({"raw": audio_array, "sampling_rate": sampling_rate}, return_language=True)

        execution_time = time.perf_counter() - start
        chunks = result.get("chunks") or []

        logger.info("Transcription automatique terminee.")
        return {
            "text": result.get("text", "").strip(),
            "language": chunks[0].get("language") if chunks else "french",
            "execution_time_seconds": execution_time,
        }