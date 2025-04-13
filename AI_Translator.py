from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import whisper
import tempfile
from gtts import gTTS
from googletrans import Translator

# Initialize Flask app
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Load Whisper model
model = whisper.load_model("small")
translator = Translator()

def recognize_speech(audio_path):
    """Transcribe speech using Whisper."""
    result = model.transcribe(audio_path, temperature=0.0, beam_size=10)
    return result["text"].strip()

def translate_text(text, target_lang="fr"):
    """Translate text using Google Translate."""
    detected_lang = translator.detect(text).lang
    translated = translator.translate(text, src=detected_lang, dest=target_lang)
    return translated.text, detected_lang

def text_to_speech(text, lang="en"):
    """Convert text to speech and save it as an audio file."""
    tts = gTTS(text=text, lang=lang)
    temp_path = tempfile.mktemp(suffix=".mp3")
    tts.save(temp_path)
    return temp_path

@app.route("/")
def serve_index():
    """Serve frontend page."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve audio files."""
    return send_from_directory(tempfile.gettempdir(), filename)

@app.route("/translate", methods=["POST"])
def translate():
    """Handle audio upload, recognition, translation, and speech conversion."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Get target language from the form data
    target_lang = request.form.get("target_lang", "fr")  # Default to English if not provided
    audio_file = request.files["audio"]

    # Save the uploaded audio file to a temporary location
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    audio_file.save(temp_audio_path)

    try:
        # Recognize speech
        recognized_text = recognize_speech(temp_audio_path)
        if not recognized_text:
            return jsonify({"error": "Failed to recognize speech"}), 400

        # Translate text
        translated_text, source_lang = translate_text(recognized_text, target_lang)

        # Convert to speech
        speech_path = text_to_speech(translated_text, lang=target_lang)
        speech_filename = os.path.basename(speech_path)

        # Clean up temporary files
        os.remove(temp_audio_path)

        return jsonify({
            "success": True,
            "recognized_text": recognized_text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "audio_url": f"/audio/{speech_filename}"  # Serve the audio file
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files in case of errors
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    app.run(debug=True)