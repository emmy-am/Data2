#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import random

song_recommendations = {
    "happy": [
        "Happy - Pharrell Williams",
        "Walking on Sunshine - Katrina and the Waves",
        "Can’t Stop the Feeling - Justin Timberlake",
        "Shake It Off - Taylor Swift",
        "ludacris -  get back"
    ],
    "sad": [
        "Someone Like You - Adele",
        "Fix You - Coldplay",
        "Let Her Go - Passenger",
        "Fade Into You"
    ],
    "neutral": [
        "Bohemian Rhapsody - Queen",
        "Hotel California - Eagles",
        "Wonderwall - Oasis",
        "Imagine - John Lennon",
        
    ],
    "uplifting": [
        "Don’t Stop Believin’ - Journey",
        "Stronger - Kelly Clarkson",
        "Eye of the Tiger - Survivor",
        "I Will Survive - Gloria Gaynor",
        "Can't Stop the Feeling",
        "Say little prayer for you- Aretha Franklin",
        "Imagine Dragons - Thunder"
        
    ]
}

def get_song_recommendation(mood):
    if mood == "sad":
        sad_song = random.choice(song_recommendations["sad"])
        uplifting_song = random.choice(song_recommendations["uplifting"])
        return (
            f" Here's a song to match your mood: {sad_song}\n"
            f"If you'd like to cheer up, try this uplifting song: {uplifting_song}"
        )
    elif mood in song_recommendations:
        suggested_song = random.choice(song_recommendations[mood])
        return f"Here's a song for you: {suggested_song}"
    else:
        return "I'm not sure about that mood. Try happy, sad, or neutral."

st.title("🎵 Mood-Based Song Recommender")

user_mood = st.text_input("Enter your mood (e.g., happy, sad, neutral):")

if st.button("Get Song Recommendation"):
    if user_mood.lower() in ["happy", "sad", "neutral"]:
        recommendation = get_song_recommendation(user_mood.lower())
        st.write(recommendation)
    else:
        st.write("Sorry, I don't recognize that mood. Try 'happy', 'sad', or 'neutral'.")


# In[ ]:




