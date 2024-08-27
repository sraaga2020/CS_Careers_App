#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sklearn as sk
from fuzzywuzzy import process
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

# create X and y variables
filtered_df = pd.read_csv('career_skills_dataset_with_pattern.csv')
X = filtered_df.drop(columns=['Career','EncodeCareer'])
y = filtered_df['EncodeCareer']

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# define skill search by career function 
st.title("Career Predictor Based on Skills")
def skillsByCareer():
    global filtered_df
    chosen_career = st.text_input("What career would you like to find skills for? ").strip().lower()
    best_match, score = process.extractOne(chosen_career, filtered_df['Career'].str.strip().values)
    if chosen_career:
        if score < 70:
            st.write("Career not found. Please try a different career.")
        else:
            chosen_career = best_match
            st.write(f"Skills and proficiencies required for {chosen_career}:")
            filtered_df = filtered_df.query('Career == @chosen_career')
            filtered_df = filtered_df.drop(columns=['Career','EncodeCareer'])
            top_skills = filtered_df.mean().nlargest(5)
            top_skills_df = top_skills.reset_index()
            top_skills_df.columns = ['Skill', 'Average Proficiency']
            top_skills_df.index = top_skills_df.index + 1
            st.dataframe(top_skills_df)



# define career search by skills function
def careerBySkills():
    skill_set = []
    proficiency_set = []
    global filtered_df
    skills = filtered_df.columns[:-2]
    st.write("Choose your most proficient CS skills.") 
    numSkills = st.number_input("How many skills would you like to enter? (The rest of the skills will be assumed as average or novice proficiency.)")
    if numSkills % 1 != 0 or numSkills < 1:
        st.write("Please enter an integer number of skills.")
    else:
        numSkills = int(numSkills)
        for i in range(numSkills):
            skill = st.text_input(f"Skill {i+1}: ").strip()
            if skill: 
                stripped_skills = [s.strip() for s in skills]
                best, score = process.extractOne(skill, stripped_skills)
                if score < 70:
                    st.write("Skill not found. Please try a different skill.")
                else:
                    print(best)
                    skill_set.append(best)
                    proficiency = st.slider(f"Proficiency {i+1}:", 1, 10, 1)
                    proficiency_set.append(proficiency)

        if numSkills == 0 or len(skill_set)!=numSkills or len(proficiency_set)!=numSkills:
            st.write("Please enter the chosen number skills and proficiencies.")     
        else: 
            existing_skills_set = set([s.strip() for s in skills])
            for skill in existing_skills_set:
                if skill not in skill_set:
                    skill_set.append(skill)
                    proficiency_set.append(random.randint(1, 5))
            skills_with_prof = dict(zip(skill_set, proficiency_set))
            X_new_df = pd.DataFrame([skills_with_prof], columns=skills)
            prediction = dtree.predict(X_new_df)
        
            
            for value in filtered_df['EncodeCareer'].unique():
                if prediction[0] == value:
                    st.title(f"Predicted Career: {filtered_df[filtered_df['EncodeCareer'] == value]['Career'].values[0]}")
                


# define main function
def main():
    skillsByCareer()

if __name__ == "__main__":
    main()