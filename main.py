from jobs.stt_jobs.stt import speech_to_text
from jobs.gen_jobs.gen import generate_text

if __name__ == "__main__":
    #escolha 1 = Vosk, escolha 2 = Whisper, escolha 3 = Whisper Realtime, escolha 4 = Input por teclado
    text = speech_to_text(escolha="3")
    print(f"Texto reconhecido: {text}")
    
    