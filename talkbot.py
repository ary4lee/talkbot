import copy
import enum
import llama_cpp
import numpy
import os
import pyttsx3
import queue
import re
import speech_recognition
import threading
import torch
import warnings
import whisper
import sys

# !!! IMPORTANT: THIS PROGRAM REQUIRES THE LLAMA GGUF MODEL FROM HUGGINGFACE
# DOWNLOAD IT HERE: https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
# WHEN DOWNLOADED, CHANGE THE VARIABLE BELOW TO THE PATH OF THE LOCATION OF YOUR MODEL
LOCAL_MODEL_PATH = "CHANGE TO THE PATH OF YOUR MODEL"


class Message:
    Category = enum.Enum('Category', ['DATA', 'END', 'EXIT'])

    content = None
    category = Category.DATA

    # constructor
    def __init__(self, content, category):
        self.content = content
        self.category = category


warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


audio_queue = queue.Queue()
question_queue = queue.Queue()
speech_queue = queue.Queue()


# ASR setup
SAMPLE_RATE = 16000
PHRASE_DURATION = 15 # seconds
ASR_MODEL = "small"
ASR_LANG = "en"
MIC_WORDS = "Calibration time:"
ECHO_WORDS = "You Said:"
WAKE_WORDS = ["Hey, talkbot", "Hey, talk bot", "Hey talk bot", "Hey talkbot", "Hey, talk about", "Hey talk about"]
WAKE_REPLY = "What question do you have?"
MIN_PHRASE = 4
EXIT_WORDS = ["Exit Talk Bot", "Exit Talkbot", "Exit hotpot", "A talkbot", "A talk bot", "Exit pop pop", "Exit pop-up"]
EXIT_REPLY = "Exited."
def recognizer_callback(_, audio:speech_recognition.AudioData) -> None:
    raw_data = audio.get_raw_data()
    audio_queue.put(raw_data)
recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone(sample_rate=SAMPLE_RATE)
recognizer.listen_in_background(microphone, recognizer_callback, phrase_time_limit=PHRASE_DURATION)
model = whisper.load_model(ASR_MODEL)


def get_transcript(audio_queue):
    # Convert in-ram buffer to something the model can use directly without needing a temp file.
    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    audio_data = audio_queue.get(block=True)
    audio_data = numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0
    audio_data = whisper.pad_or_trim(audio_data, length=SAMPLE_RATE*PHRASE_DURATION)

    transcribe_result = model.transcribe(audio_data, language=ASR_LANG, fp16=torch.cuda.is_available())
    transcribe_text = transcribe_result['text'].strip()
    print(ECHO_WORDS, transcribe_text)
    return transcribe_text

def reset_transcript(audio_queue):
    audio_queue.queue.clear()


# TTS setup

def TTSProcessor(speechtext):
    print("TTS Active!")
    TTS_RATE = 175
    TTS_LANG = "en_US"
    tts = pyttsx3.init()
    tts.setProperty("rate", TTS_RATE)
    voices = tts.getProperty("voices")
    for voice in voices:
        if voice.languages[0] == TTS_LANG:
            tts.setProperty("voice", voice.id)
   
    """ speech = speech_queue.get(block=True)

    if speech.category == Message.Category.EXIT:
        print("TTS stopped...")
        done = True
    elif speech.category == Message.Category.END:
        reset_transcript(audio_queue)
    """
    print("TTS Running!")
    tts.say(speechtext)
    tts.runAndWait()
    reset_transcript(audio_queue)
 

# LLM setup
INPUT_LIMIT = 8192
OUTPUT_LIMIT = 2048
def LLMProcessor(question_queue, speech_queue):
    SYSTEM_PROMPT = """
You are a helpful AI assistant called Talkbot. Try to answer the question as best as you can.
"""
    PROMPT_TEMPLATE = """
<|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    while True:
        question = question_queue.get(block=True) # Gets the question from the queue

        if question.category == Message.Category.EXIT: # If the question is to exit...
            speech_queue.put(Message(None, Message.Category.EXIT))
            break

        prompt = PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, question=question.content)
        # print(prompt)
        print ("Processing...")

        llm = llama_cpp.Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=INPUT_LIMIT,
            n_batch=128, # Should be between 1 and n_ctx
            verbose=False
        )
        output = llm(
            prompt=prompt,
            max_tokens=OUTPUT_LIMIT,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["#"],
            stream=False
        )

        output_text = copy.deepcopy(output)["choices"][0]["text"]
        print(output_text, end ="")
        TTSProcessor(output_text)
llm = threading.Thread(target=LLMProcessor, args=(question_queue, speech_queue, ))
llm.start()

# TTS setup


done = False
awake = False
while not done:
    # listen to microphone and transcribe with whisper
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 0
    recognizer.adjust_for_ambient_noise(microphone, duration=1)
    print(MIC_WORDS, recognizer.energy_threshold)
    print("Start Speaking:")
    
    while True:
        transcribe_text = get_transcript(audio_queue)
        if not awake:
            if any(sub in transcribe_text for sub in WAKE_WORDS):
                TTSProcessor(WAKE_REPLY)
                speech_queue.put(Message(WAKE_REPLY, Message.Category.DATA))
                speech_queue.put(Message(None, Message.Category.END))
                awake = True
                break
        else:
            if any(sub in transcribe_text for sub in EXIT_WORDS):
                print("Exiting Talkbot...")
                speech_queue.put(Message(EXIT_REPLY, Message.Category.DATA))
                question_queue.put(Message(None, Message.Category.EXIT))
                awake = False
                sys.exit()
                break
            elif len(transcribe_text) > MIN_PHRASE:
                speech_text = "You asked, " + transcribe_text
                print(speech_text.replace(WAKE_REPLY, ""))
                TTSProcessor(speech_text.replace(WAKE_REPLY, ""))
                speech_queue.put(Message(speech_text, Message.Category.DATA))
                question_queue.put(Message(transcribe_text, Message.Category.DATA))
                awake = False
                break

    # get speech text and say it with tts
    # typical values for a silent room are 0 to 100
    # typical values for speaking are between 150 and 3500
    # sensitive microphone or microphones in louder rooms might have up to 4000
    recognizer.energy_threshold = 4000000 # increase to avoid listening to tts

