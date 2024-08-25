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

# restore original form of 'Career' column
#filtered_df['Career'] = label_encoder.inverse_transform(filtered_df['Career'])

# define skill search by career function 
st.title("Career Predictor Based on Skills")
def skillsByCareer():
    global filtered_df
    chosen_career = st.text_input("What career would you like to find skills for? ").strip().lower()
    best_match, score = process.extractOne(chosen_career, filtered_df['Career'].str.strip().values)
    chosen_career = best_match
    st.write(f"Skills and proficiencies required for {chosen_career}:")
    filtered_df = filtered_df.query('Career == @chosen_career')
    filtered_df = filtered_df.drop(columns=['Career','EncodeCareer'])
    top_skills = filtered_df.mean().nlargest(5)
    st.write(top_skills)




skill_set = []
proficiency_set = []
# define career search by skills function
def careerBySkills():
    global filtered_df
    skills = filtered_df.columns[:-2]
    global skill_set
    global proficiency_set
    st.write("Choose your most proficient CS skills.") 
    numSkills = int(st.number_input("How many skills would you like to enter? (The rest of the skills will be assumed as average or novice proficiency.)"))
    for i in range(numSkills):
        skill = st.text_input(f"Skill {i+1}: ").strip()
        stripped_skills = [s.strip() for s in skills]
        best, score = process.extractOne(skill, stripped_skills)
        print(best)
        skill_set.append(best)
        proficiency = st.slider(f"Proficiency {i+1}:", 1, 10, 5)
        proficiency_set.append(proficiency)

    existing_skills_set = set([s.strip() for s in skills])
    for skill in existing_skills_set:
        if skill not in skill_set:
            skill_set.append(skill)
            proficiency_set.append(random.randint(1, 5))
    skills_with_prof = dict(zip(skill_set, proficiency_set))
    X_new_df = pd.DataFrame([skills_with_prof], columns=skills)
    prediction = dtree.predict(X_new_df)
    '''
    row = filtered_df[filtered_df['EncodeCareer'] == prediction]
    first_row = row.iloc[0]
    # Retrieve another column's value from that row
    career_predict = first_row['Career']
    '''
    st.write(prediction)



# define main function
def main():
    skillsByCareer()

if __name__ == "__main__":
    main()