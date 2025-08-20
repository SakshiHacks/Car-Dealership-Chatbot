import pyttsx3
import nltk
import numpy as np
import random
import speech_recognition as sr
import pickle
import json
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()


# Load data and model
intents = json.loads(open("/content/intents.json").read())
words = pickle.load(open("/content/words.pkl", "rb"))
classes = pickle.load(open("/content/classes.pkl", "rb"))
model = load_model("/content/chatbot_model.h5")

# Setup TTS
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)  # Change index for different voices

def speak(text):
    engine.say(text)
    engine.runAndWait()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm not sure how to help with that."

def start_chat():
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
    except Exception as e:
        print("Microphone initialization failed:", str(e))
        speak("Microphone not found. Switching to text input.")
        mic = None

    print("AutoBot is running.")
    speak("Hello! I am AutoBot, your car dealership assistant. How can I help you today?")

    while True:
        if mic:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                speak("I'm listening.")
                try:
                    audio = recognizer.listen(source)
                    user_input = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    print("Sorry, I did not catch that.")
                    speak("Sorry, I did not catch that.")
                    continue
                except Exception as e:
                    print("Microphone Error:", str(e))
                    speak("Microphone error. Try again.")
                    continue
        else:
            user_input = input("You (type input): ")

        print("You said:", user_input)
        speak("You said: " + user_input)
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
        print("AutoBot:", response)
        speak(response)

if __name__ == "__main__":
    start_chat()
