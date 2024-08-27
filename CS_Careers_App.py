#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import random
from sklearn.model_selection import train_test_split
import streamlit as st
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# number of row
num_rows = 1000

# list of skills
skills = [
    'Programming', 'Data Analysis', 'Machine Learning', 'Artificial Intelligence', 
    'Database Management', 'Software Development', 'Cybersecurity', 'Cloud Computing', 
    'Networking', 'Web Development', 'JavaScript', 'Python', 'Java', 'C++', 
    'SQL', 'HTML/CSS', 'APIs', 'Version Control', 'Agile Methodologies', 
    'DevOps', 'Data Visualization', 'Big Data', 'Linux', 'Windows', 
    'Project Management', 'Systems Design', 'Embedded Systems', 'Algorithms', 
    'Data Structures', 'Mobile Development', 'UX/UI Design', 'Testing/QA', 
    'Blockchain', 'IT Support', 'Business Analysis', 'Game Development', 
    'Cloud Services', 'Software Architecture', 'Scripting', 'Technical Documentation', 
    'Ethical Hacking', 'IoT', 'Virtual Reality', 'Augmented Reality', 
    'System Administration', 'Business Intelligence', 'Quantitative Analysis', 
    'Data Engineering', 'Computer Vision', 'Natural Language Processing', 
    'Robotics', 'Information Retrieval', 'Software Engineering', 'Network Security', 
    'Computer Graphics', 'Embedded Programming', 'Data Warehousing', 
    'Information Systems', 'Technical Support', 'Product Management', 
    'Research & Development', 'Digital Marketing', 'Algorithm Design', 
    'Application Development', 'Infrastructure Management', 'Enterprise Solutions'
]

# list of careers
careers_list = [
    'Software Engineer', 'Data Scientist', 'Systems Analyst', 'Web Developer', 
    'Database Administrator', 'Network Engineer', 'Cybersecurity Analyst', 
    'Machine Learning Engineer', 'Cloud Architect', 'DevOps Engineer', 
    'Front-End Developer', 'Back-End Developer', 'Full Stack Developer', 
    'AI Research Scientist', 'Game Developer', 'Mobile App Developer', 
    'IT Support Specialist', 'Business Intelligence Analyst', 'Embedded Systems Engineer', 
    'IT Project Manager', 'UX/UI Designer', 'Technical Writer', 'Blockchain Developer', 
    'IT Consultant', 'Data Engineer', 'Quantitative Analyst', 'Data Architect', 
    'Systems Administrator', 'Site Reliability Engineer', 'AR/VR Developer', 
    'Ethical Hacker', 'Digital Marketer', 'Software Development Manager', 
    'Product Manager', 'Network Security Engineer', 'IT Auditor', 
    'Technical Support Specialist', 'Algorithm Engineer', 'Applications Developer', 
    'Research Scientist', 'Computer Vision Engineer'
]

# skills needed for each career

career_skills = {
    'Software Engineer': ['Programming', 'Software Development', 'Version Control', 'Algorithms', 'Data Structures'],
    'Data Scientist': ['Data Analysis', 'Machine Learning', 'Data Visualization', 'Big Data', 'Python'],
    'Systems Analyst': ['Business Analysis', 'Project Management', 'Database Management', 'Systems Design', 'Technical Documentation'],
    'Web Developer': ['Web Development', 'JavaScript', 'HTML/CSS', 'APIs', 'Version Control'],
    'Database Administrator': ['Database Management', 'SQL', 'Data Warehousing', 'Systems Administration', 'Technical Support'],
    'Network Engineer': ['Networking', 'Network Security', 'Cloud Computing', 'Systems Administration', 'Project Management'],
    'Cybersecurity Analyst': ['Cybersecurity', 'Network Security', 'Ethical Hacking', 'Systems Administration', 'Technical Support'],
    'Machine Learning Engineer': ['Machine Learning', 'Data Analysis', 'Algorithms', 'Data Structures', 'Programming'],
    'Cloud Architect': ['Cloud Computing', 'Cloud Services', 'Systems Design', 'Software Architecture', 'Networking'],
    'DevOps Engineer': ['DevOps', 'Cloud Computing', 'Software Development', 'Version Control', 'Project Management'],
    'Front-End Developer': ['Web Development', 'JavaScript', 'HTML/CSS', 'UI/UX Design', 'Version Control'],
    'Back-End Developer': ['Programming', 'Database Management', 'APIs', 'Software Development', 'Version Control'],
    'Full Stack Developer': ['Programming', 'Web Development', 'Database Management', 'APIs', 'Version Control'],
    'AI Research Scientist': ['Artificial Intelligence', 'Machine Learning', 'Data Analysis', 'Algorithms', 'Data Visualization'],
    'Game Developer': ['Game Development', 'Programming', 'Software Development', 'Graphics', 'Algorithm Design'],
    'Mobile App Developer': ['Mobile Development', 'Programming', 'UI/UX Design', 'Software Development', 'Testing/QA'],
    'IT Support Specialist': ['IT Support', 'Technical Support', 'Systems Administration', 'Networking', 'Database Management'],
    'Business Intelligence Analyst': ['Business Analysis', 'Data Analysis', 'Data Visualization', 'Big Data', 'Database Management'],
    'Embedded Systems Engineer': ['Embedded Systems', 'Programming', 'Systems Design', 'Algorithms', 'Data Structures'],
    'IT Project Manager': ['Project Management', 'Business Analysis', 'Systems Design', 'Technical Documentation', 'Team Management'],
    'UX/UI Designer': ['UX/UI Design', 'Web Development', 'Technical Documentation', 'Graphics', 'User Research'],
    'Technical Writer': ['Technical Documentation', 'Software Development', 'Business Analysis', 'Project Management', 'Technical Support'],
    'Blockchain Developer': ['Blockchain', 'Programming', 'Cryptography', 'Software Development', 'Networking'],
    'IT Consultant': ['Business Analysis', 'Project Management', 'Systems Design', 'Technical Documentation', 'Technical Support'],
    'Data Engineer': ['Data Engineering', 'Data Analysis', 'Big Data', 'Database Management', 'SQL'],
    'Quantitative Analyst': ['Quantitative Analysis', 'Data Analysis', 'Programming', 'Statistics', 'Financial Modelling'],
    'Data Architect': ['Data Architecture', 'Database Management', 'Data Warehousing', 'SQL', 'Data Modeling'],
    'Systems Administrator': ['Systems Administration', 'Networking', 'Technical Support', 'Cloud Computing', 'Database Management'],
    'Site Reliability Engineer': ['Site Reliability', 'Systems Administration', 'Cloud Computing', 'Software Development', 'Monitoring'],
    'AR/VR Developer': ['Augmented Reality', 'Virtual Reality', 'Programming', '3D Modeling', 'Software Development'],
    'Ethical Hacker': ['Ethical Hacking', 'Cybersecurity', 'Network Security', 'Systems Administration', 'Penetration Testing'],
    'Digital Marketer': ['Digital Marketing', 'SEO', 'Content Creation', 'Social Media', 'Analytics'],
    'Software Development Manager': ['Software Development', 'Project Management', 'Team Management', 'Technical Documentation', 'Product Management'],
    'Product Manager': ['Product Management', 'Project Management', 'Business Analysis', 'Technical Documentation', 'Marketing'],
    'Network Security Engineer': ['Network Security', 'Cybersecurity', 'Ethical Hacking', 'Systems Administration', 'Technical Support'],
    'IT Auditor': ['IT Auditing', 'Systems Administration', 'Compliance', 'Technical Support', 'Risk Management'],
    'Technical Support Specialist': ['Technical Support', 'IT Support', 'Customer Service', 'Systems Administration', 'Troubleshooting'],
    'Algorithm Engineer': ['Algorithm Design', 'Programming', 'Data Structures', 'Machine Learning', 'Software Development'],
    'Applications Developer': ['Application Development', 'Programming', 'Software Development', 'UI/UX Design', 'Testing/QA'],
    'Research Scientist': ['Research & Development', 'Data Analysis', 'Scientific Computing', 'Programming', 'Technical Documentation'],
    'Computer Vision Engineer': ['Computer Vision', 'Machine Learning', 'Programming', 'Data Analysis', 'Algorithms']
}

# initialize dataframe
df = pd.DataFrame(columns=skills + ['Career'])

# generate random data
np.random.seed(0)
for i in range(num_rows):
    career = np.random.choice(careers_list)

    # random proficiency for normal skills
    proficiency = np.random.randint(1, 6, size=len(skills))
    
    # higher proficiency for career skills
    if career in career_skills:
        skill_indices = [skills.index(skill) for skill in career_skills[career] if skill in skills]
        proficiency[skill_indices] = np.random.randint(8, 11, size=len(skill_indices))
    
    df.loc[i] = list(proficiency) + [career]

# encode careers
filtered_df = df.copy()
career_col = 'Career'
label_encoder = LabelEncoder()
filtered_df[career_col] = label_encoder.fit_transform(filtered_df[career_col])

# save to csv
filtered_df.to_csv('career_skills_dataset_with_pattern.csv', index=False)


filtered_df = pd.read_csv('career_skills_dataset_with_pattern.csv')

# create X and y variables
X = filtered_df.drop(columns='Career')
y = filtered_df['Career']

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# restore original form of 'Career' column
filtered_df['Career'] = label_encoder.inverse_transform(filtered_df['Career'])

# define skill search by career function 
st.title("Career Predictor Based on Skills")
def skillsByCareer():
    global filtered_df
    chosen_career = st.text_input("What career would you like to find skills for? ").strip().lower()
    best_match, score = process.extractOne(chosen_career, df['Career'].str.strip().values)
    chosen_career = best_match
    st.write(f"Skills and proficiencies required for {chosen_career}:")
    filtered_df = filtered_df.query('Career == @chosen_career')
    filtered_df = filtered_df.drop(columns=['Career'])
    top_skills = filtered_df.mean().nlargest(5)
    st.write(top_skills)




skill_set = []
proficiency_set = []
# define career search by skills function
def careerBySkills():
    global filtered_df
    global skills
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
    original_prediction = label_encoder.inverse_transform([prediction])[0]
    st.write(original_prediction)



# define main function
def main():
    skillsByCareer()

if __name__ == "__main__":
    main()