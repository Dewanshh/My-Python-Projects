# Customer Feedback Analysis


from deepface import DeepFace

from PIL import Image

import cv2

import matplotlib.pyplot as plt
import speech_recognition as sr
from textblob import TextBlob
 

img_path='happy1.jpeg'

img=cv2.imread(img_path)

demography =DeepFace.analyze(img_path)


r = sr.Recognizer()

mic = sr.Microphone(device_index=1)

with sr.Microphone() as source:

    r.adjust_for_ambient_noise(source, duration=1)
    print()
    print("For which product you want to give review: ")
    print("listening...")
    audio = r.listen(source, timeout=3)
    print("Time's up, Thanks!!")

try:
    product = r.recognize_google(audio)
    print("Text:" + r.recognize_google(audio))

except:
    print("unable to understand")
    sys.exit()

with sr.Microphone() as source:

    r.adjust_for_ambient_noise(source, duration=1)
    print()

    print("Please, give the review: ")
    print("listening...")
    audio = r.listen(source, timeout=5)
    print("Time's up, Thanks!!")

try:
    text = r.recognize_google(audio)
    print("Text:" + r.recognize_google(audio))
except:
    print("unable to understand")
    sys.exit()

print()

obj = TextBlob(text)

sentiment, subjectivity = obj.sentiment

print(obj.sentiment)
print("the review is about: ", product)
print("Given by person of: ")
print("Age: ",demography["age"])
print("gender:", demography["gender"])
print("Emotion:", demography["dominant_emotion"]) 
print("race:", demography["dominant_race"])
plt.imshow
image=Image.open(img_path)

image.show()
if sentiment == 0:
    print('   The review is :  neutral')
elif sentiment > 0:
    print('   The review is  : positive')
else:
    print('   The review is  : negative')





