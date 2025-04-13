csv_path = "/home/kushal/Documents/Code/Current-Projects/Akinator/anime_traits_better.csv"

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(csv_path)

# This is for encoding genders
label_encoder = LabelEncoder()
df["Gender_encoded"] = label_encoder.fit_transform(df["Gender"])
gender_mapping = label_encoder.classes_
print("Gender mapping:", gender_mapping)

male_df = df[df["Gender_encoded"] == gender_mapping.tolist().index("Male")]
female_df = df[df["Gender_encoded"] == gender_mapping.tolist().index("Female")]
neutral_df = df[df["Gender_encoded"] == gender_mapping.tolist().index("Nuetral")]
print(f"Male characters: {len(male_df)}")
print(f"Female characters: {len(female_df)}")
print(f"Neutral characters: {len(neutral_df)}")

# This thing is for preparaing and training
def prepare_and_train_model(gender_df):
    # remove unnecessary features from x
    X = gender_df.drop(["Names", "Id", "Gender", "Gender_encoded"], axis=1)
    y = gender_df["Names"]
    
    # Encode Hair_Color
    label_encoder = LabelEncoder()
    X["Hair_Color"] = label_encoder.fit_transform(X["Hair_Color"])
    hair_mapping = label_encoder.classes_
    
    # Finding information gain 
    info_gain = {}
    for feature in X:
        yes_count = np.sum(X[feature] == 1)
        no_count = np.sum(X[feature] == 0)
        
        total = yes_count + no_count
        p_yes = yes_count / total if total > 0 else 0
        p_no = no_count / total if total > 0 else 0
        
        h_yes = -p_yes * np.log2(p_yes) if p_yes > 0 else 0
        h_no = -p_no * np.log2(p_no) if p_no > 0 else 0
        
        h_split = p_yes * h_yes + p_no * h_no
        info_gain[feature] = h_split
    
    # Sorting in descending order
    sorted_features = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)
    
    for feature, ig in sorted_features:
        if ig < 0.06:
            X = X.drop(feature, axis=1)
    
    model = KNeighborsClassifier(n_neighbors=2,weights='distance') # Actually we use "weighted" knn here
    model.fit(X, y)
    return model, X.columns, hair_mapping, sorted_features

# Training for all three genders
male_model, male_features, male_hair_mapping, male_sorted_features = prepare_and_train_model(male_df)
female_model, female_features, female_hair_mapping, female_sorted_features = prepare_and_train_model(female_df)
neutral_model, neutral_features, neutral_hair_mapping, neutral_sorted_features = prepare_and_train_model(neutral_df)

print("All models trained successfully.")

# MAIN function, this is what does the whole user input thingy
def predict_character():
    print("\nx--- ANIME CHARACTER GUESSER ---x")
    print("What gender is your character?")
    for i, gender in enumerate(gender_mapping):
        print(f"{i}: {gender}")
    
    # Getting gender as either male,female or neutral
    while True:
        try:
            gender_choice = int(input("Enter the number for your character's gender: "))
            if 0 <= gender_choice < len(gender_mapping):
                selected_gender = gender_mapping[gender_choice]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Taking the right model
    if selected_gender == "Male":
        model = male_model
        features = male_features
        hair_mapping = male_hair_mapping
        sorted_features_list = male_sorted_features
    elif selected_gender == "Female":
        model = female_model
        features = female_features
        hair_mapping = female_hair_mapping
        sorted_features_list = female_sorted_features
    else:  # For Neutral
        model = neutral_model
        features = neutral_features
        hair_mapping = neutral_hair_mapping
        sorted_features_list = neutral_sorted_features
    
    print(f"\nUsing {len(features)} features for {selected_gender} characters.") # Shows how many featues have ig >= 0.06 for the chosen gender
    
    input_from_user = {}  # Put all input in this thing
    
    for feature in features:
        if feature == "Hair_Color":
            print("\nHair color options:")
            for i, color in enumerate(hair_mapping):
                print(f"{i}: {color}")
            
            while True:
                try:
                    value = int(input("Enter the number for your character's hair color: "))
                    if 0 <= value < len(hair_mapping):
                        input_from_user[feature] = value
                        break
                    else:
                        print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            # This is for the rest of the features
            response = input(f"Does your character have/is {feature.lower()}? (y/n): ")
            input_from_user[feature] = 1 if response.lower() in ['y', 'yes'] else 0 # we put 1 or 0 whenever user says y or n.
    
    # convert array to a format to feed to ml model.
    input_array = np.zeros((1, len(features)))
    for i, feature in enumerate(features):
        if feature in input_from_user:
            input_array[0, i] = input_from_user[feature]
    
    # Make prediction
    predicted_probs = model.predict_proba(input_array)[0] # put predict_proba because we need probability estimates for each class to rank the most likely characters
    class_labels = list(model.classes_)
    
    # Top 2 predicted characters
    top_indices = np.argsort(predicted_probs)[-2:][::-1]
    
    print("\nx--- PREDICTION RESULTS ---x")
    print(f"Most likely character: {class_labels[top_indices[0]]} ({predicted_probs[top_indices[0]] * 100:.2f}%)")    
    print("\nTop 2 matches:")
    for i, idx in enumerate(top_indices):
        char_name = class_labels[idx]
        char_prob = predicted_probs[idx] * 100
        print(f"{i+1}. {char_name}: {char_prob:.2f}%")


predict_character()

