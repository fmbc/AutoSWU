#----------------------------------------------------------------------------
# Project Name : Analysis of behavior in the vehicle using wireless network
# Author : AutoSWU (Jihye Shin  / Sohee Won / Seorin Jung / MinYoung Choi)
# Data : 2023.02.03
#----------------------------------------------------------------------------

# Use wav files for visualization and analysis.
import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

# Used for piezo buzzer
import RPi.GPIO as GPIO
import time

# Using for text transmission
import datetime
from twilio.rest import Client

FIG_SIZE = (15,10)

file = "head_data.wav"

sig, sr = librosa.load(file, sr=22050)

print(sig,sig.shape)

# Wavefrom Visualization
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(sig, sr, alpha=0.5)
plt.xlabel("Time (s)")                     
plt.ylabel("Amplitude")                     
plt.title("Waveform")                        
plt.xlim(0,4)                               
plt.ylim(-0.075, 0.06)        

# AES 
from Crypto.Cipher import AES
from secrets import token_bytes

key = token_bytes(16)

def encrypt(msg):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(msg.encode('utf8'))
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    try:
        cipher.verify(tag)
        return plaintext.decode('utf8')
    except:
        return False

#서울여자대학교 제2 과학관의 위도는 37.62801310620607이고 경도는 127.09137429778716 입니다
nonce, ciphertext, tag = encrypt('The latitude of Seoul Women`s University Science Hall 2 is 37.62801310620607 and the longitude is 127.09137429778716.')
plaintext = decrypt(nonce, ciphertext, tag)
if not plaintext:
    print('Message is corrupted')                


# body abnormality detection
if sr>= 0.04:                                 # Detects abnormalities in the body at a specific frequency
  print("\n*****[AutoSWU] detected!!*****\n")    # Output that it was detected
  print(f'driver GPS information: {ciphertext}')
 
  # Warning Sound Output
  buzzer = 18
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(buzzer, GPIO.OUT)
  GPIO.setwarnings(False)
  
  pwn=GPIO.PWM(buzzer, 262)
  pwn.start(50.0)
  time.sleep(1.5)
  
  pwn.stop()
  GPIO.cleanup()

  # Text Transfer Part
  account_sid = 'AC119bdb4229d503ede749ebff8cbfe2f6'
  auth_token = '8955b71ac558117ba5d57832cc3cc1bb'
  client = Client(account_sid, auth_token)
      
  message = client.messages.create(
      to="+8201036233002",
      from_="+12183044999",
      body="\n[AutoSWU]\n Driver's body abnormality detected!!\n")

  print(message.sid, datetime.datetime.now())

# FFT(Fast Fourier Transform) : Identify and visualize the amount of frequency.
fft = np.fft.fft(sig)

# Find 'magnitude' as the absolute value of the complex space value
magnitude = np.abs(fft) 

# Create Frequency Value
f = np.linspace(0,sr,len(magnitude))

# The 'spectrum' that passes through the Fourier transform comes out in a symmetrical structure and uses only the front half to fly half of the 'high frequency' part
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.ylim([0, 150])

# STFT(Short-Time Fourier Transform)
hop_length = 512          
n_fft = 2048              

hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(stft)
 
log_spectrogram = librosa.amplitude_to_db(magnitude)

# STFT Visualization
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

# MFCC(Mel Frequency Cepstral Coefficient)
MFCCs = librosa.feature.mfcc(sig, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# MFCCs Visualization
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

plt.show()
