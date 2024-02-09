import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
def calculate_subject_stats_marks(dataset, year, marks_key):
    subject_stats_marks = {}
    for subject in range(6):
        subject_marks = []
        for details in dataset[year].values():
            marks = details[marks_key]
            if marks[subject] != '0':
                subject_marks.append(float(marks[subject]))
            else:
                subject_marks.append(50)
        
        avg_marks = np.mean(subject_marks)
        sd_marks = np.std(subject_marks, ddof=1) if len(subject_marks) > 1 else 0
        paper_toughness = 15 if avg_marks >= 90 else (10 if 75 <= avg_marks <= 89 else 5)
        
        subject_stats_marks[subject] = (avg_marks, sd_marks, paper_toughness)
    
    return subject_stats_marks
def count_pass_fail_subjects_gender_hostel(dataset):
    pass_fail_counts = {}
    pass_threshold = 50  

    for subject in range(6):
        pass_fail_counts[subject] = {
            'Day Scholar Boys': {'Pass': 0, 'Fail': 0},
            'Day Scholar Girls': {'Pass': 0, 'Fail': 0},
            'Hostel Boys': {'Pass': 0, 'Fail': 0},
            'Hostel Girls': {'Pass': 0, 'Fail': 0}
        }

    for class_name, subjects in dataset.items():
        if class_name == "oneyr":  
            for subject_id, details in subjects.items():
                marks_present = any(mark != '0' for mark in details['marks1'])

                if marks_present:
                    predicted_marks = details.get('marks4', None)
                    if not predicted_marks:
                        predicted_marks = details.get('marks3', None)
                    if not predicted_marks:
                        predicted_marks = details.get('marks2', None)

                    if predicted_marks:
                        for subject, mark in enumerate(predicted_marks):

                            category = f"{details['hord']} {'Boys' if details['gen'] == 'M' else 'Girls'}"

                            if details['hord'] == 'D':
                                category = f"Day Scholar {'Girls' if details['gen'] == 'F' else 'Boys'}"

                            if details['hord'] == 'H':
                                category = f"Hostel {'Girls' if details['gen'] == 'F' else 'Boys'}"

                            if mark != 'NA':
                                mark_float = float(mark)
                                if mark_float >= pass_threshold:
                                    pass_fail_counts[subject][category]['Pass'] += 1
                                else:
                                    pass_fail_counts[subject][category]['Fail'] += 1

    return pass_fail_counts

def plot_pass_fail_subjects_gender_hostel_bar_graph(pass_fail_counts):
    subjects = list(pass_fail_counts.keys())

    bar_width = 0.2
    index = np.arange(len(subjects))

    categories = ['Day Scholar Boys', 'Day Scholar Girls', 'Hostel Boys', 'Hostel Girls']
    colors = ['blue', 'orange', 'green', 'red']

    plt.figure(figsize=(12, 8))

    for i, category in enumerate(categories):
        pass_counts = [pass_fail_counts[subj][category]['Pass'] for subj in subjects]
        fail_counts = [pass_fail_counts[subj][category]['Fail'] for subj in subjects]

        plt.bar(index + i * bar_width, pass_counts, bar_width, label=f'{category} Pass', color=colors[i])
        plt.bar(index + i * bar_width, fail_counts, bar_width, label=f'{category} Fail', color=colors[i], alpha=0.5)

    plt.xlabel('Subjects')
    plt.ylabel('Number of Students')
    plt.title('Pass/Fail Counts based on Subject, Gender, and Hostel Type')
    plt.xticks(index + bar_width * (len(categories) - 1) / 2, [f"Subject {i+1}" for i in subjects])
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_marks(your_actual_dataset):
    for class_name, subjects in your_actual_dataset.items():
      if class_name == "oneyr":
        for student_id, details in subjects.items():
            marks1_present = any(mark != '0' for mark in details['marks1'])
            marks2_present = any(mark != '0' for mark in details['marks2'])
            marks3_present = any(mark != '0' for mark in details['marks3'])

            if marks1_present and not marks2_present:  # Predict marks2
                marks1_filled = [float(mark) if mark != '0' else 50 for mark in details['marks1']]

                if all(mark == 50 for mark in marks1_filled):  # If all marks in marks1 are absent, set marks2 as 'NA'
                    details['marks2'] = 'NA'
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks2: NA")
                else:
                    subject_stats1 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks1')
                    X = np.column_stack([marks1_filled, [subject_stats1[i][0] for i in range(6)], [subject_stats1[i][1] for i in range(6)], [subject_stats1[i][2] for i in range(6)]])

                    model = RandomForestRegressor()
                    model.fit(X, marks1_filled)
                    predicted_marks2 = model.predict(X).tolist()
                    details['marks2'] = predicted_marks2
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks2: {predicted_marks2}")
                

            elif marks1_present and marks2_present:  # Predict marks3
                marks1_filled = [float(mark) if mark != '0' else 50 for mark in details['marks1']]
                marks2_filled = [float(mark) if mark != '0' else 50 for mark in details['marks2']]

                if all(mark == 50 for mark in marks1_filled) and all(mark == 50 for mark in marks2_filled):
                    details['marks3'] = ['NA'] * 6  # If absent in both marks1 and marks2, set marks3 as 'NA'
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks3: NA")
                else:
                    subject_stats1 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks1')
                    subject_stats2 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks2')

                    if all(mark == 50 for mark in marks2_filled):
                        marks2_filled = marks1_filled
                        subject_stats2 = subject_stats1

                    if all(mark == 50 for mark in marks1_filled):
                        marks1_filled = [50] * 6

                    X = np.column_stack([marks1_filled, marks2_filled, [subject_stats2[i][0] for i in range(6)], [subject_stats2[i][1] for i in range(6)], [subject_stats2[i][2] for i in range(6)]])

                    model = RandomForestRegressor()
                    model.fit(X, marks1_filled)
                    details['marks3'] = model.predict(X).tolist()
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks3: {details['marks3']}")

            elif not marks1_present and not marks2_present:  # Marks1 and Marks2 absent
                details['marks3'] = ['NA'] * 6
                student_name = details['name']
                print(f"{student_name}: Predicted Marks3:", details['marks3'])

            elif not marks1_present and marks2_present:  # Marks1 absent but Marks2 present
                marks2_filled = [float(mark) if mark != '0' else 50 for mark in details['marks2']]

                if all(mark == 50 for mark in marks2_filled):
                    details['marks3'] = ['NA'] * 6
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks3: NA")
                else:
                    subject_stats2 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks2')
                    marks1_filled = marks2_filled
                    subject_stats1 = subject_stats2

                    X = np.column_stack([marks1_filled, marks2_filled, [subject_stats1[i][0] for i in range(6)], [subject_stats1[i][1] for i in range(6)], [subject_stats1[i][2] for i in range(6)]])

                    model = RandomForestRegressor()
                    model.fit(X, marks1_filled)
                    details['marks3'] = model.predict(X).tolist()
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks3: {details['marks3']}")
            if marks3_present:  # Predict marks4
              for student_id, details in subjects.items():
                marks1_filled = [float(mark) if mark != '0' else 50 for mark in details['marks1']]
                marks2_filled = [float(mark) if mark != '0' else 50 for mark in details['marks2']]
                marks3_filled = [float(mark) if mark != '0' else 50 for mark in details['marks3']]
                
                if all(mark != 50 for mark in marks1_filled) and all(mark != 50 for mark in marks2_filled) and all(mark != 50 for mark in marks3_filled):
                    # When all marks1, marks2, and marks3 are present
                    subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                    X = np.column_stack([marks1_filled, marks2_filled, marks3_filled, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                    model = RandomForestRegressor()
                    model.fit(X, marks3_filled)
                    predicted_marks4 = model.predict(X).tolist()
                    details['marks4'] = predicted_marks4
                    student_name = details['name']
                    print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                else:
                    if all(mark == 50 for mark in marks1_filled) and all(mark == 50 for mark in marks2_filled) and all(mark == 50 for mark in marks3_filled):
                        details['marks4'] = ['NA'] * 6
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4:",details['marks4'])
                    elif all(mark == 50 for mark in marks2_filled) and all(mark == 50 for mark in marks3_filled):
                        # Absent in marks2 and marks3
                        avg_marks1 = [(marks1_filled[i] + marks3_filled[i]) / 2 for i in range(6)]
                        subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                        X = np.column_stack([marks1_filled, marks1_filled, avg_marks1, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                    elif all(mark == 50 for mark in marks3_filled):
                        # If all subjects are absent in marks3
                        avg_marks1_2 = [(marks1_filled[i] + marks2_filled[i]) / 2 for i in range(6)]
                        subject_stats2 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks2')

                        X = np.column_stack([marks1_filled, marks2_filled, avg_marks1_2, [subject_stats2[i][0] for i in range(6)], [subject_stats2[i][1] for i in range(6)], [subject_stats2[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                    elif all(mark == 50 for mark in marks2_filled):
                        # Absent only in marks2
                        marks2_filled = [50] * 6
                        subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                        X = np.column_stack([marks1_filled, marks2_filled, marks3_filled, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                    elif all(mark == 50 for mark in marks1_filled) and all(mark == 50 for mark in marks3_filled):
                        # Absent in marks1 and marks3
                        marks2_filled = [float(mark) if mark != '0' else 50 for mark in details['marks2']]
                        subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                        X = np.column_stack([marks2_filled, marks1_filled, marks3_filled, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                    elif all(mark == 50 for mark in marks1_filled) and all(mark == 50 for mark in marks2_filled):
                        # Absent in marks1 and marks2
                        subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                        X = np.column_stack([marks3_filled, marks1_filled, marks2_filled, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")
                    elif all(mark == 50 for mark in marks1_filled):
                        # Absent only in marks1
                        marks2_filled = [float(mark) if mark != '0' else 50 for mark in details['marks2']]
                        subject_stats3 = calculate_subject_stats_marks(your_actual_dataset, class_name, 'marks3')

                        X = np.column_stack([marks1_filled, marks2_filled, marks3_filled, [subject_stats3[i][0] for i in range(6)], [subject_stats3[i][1] for i in range(6)], [subject_stats3[i][2] for i in range(6)]])

                        model = RandomForestRegressor()
                        model.fit(X, marks3_filled)
                        predicted_marks4 = model.predict(X).tolist()
                        details['marks4'] = predicted_marks4
                        student_name = details['name']
                        print(f"{student_name}: Predicted Marks4: {predicted_marks4}")



def predict_marks_and_plot_pass_fail_gender_hostel(dataset):
    predict_marks(dataset)
    pass_fail_counts = count_pass_fail_subjects_gender_hostel(dataset)
    plot_pass_fail_subjects_gender_hostel_bar_graph(pass_fail_counts)


actual_dataset = {
    "oneyr":{
            "CB22001":{
                "name":"Etvg",
                "gen":"M",
                "hord":"D",
                "marks1":['70','80','90','55','60','0'],
                "marks2":['80','80','80','80','80','80'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
            "CB22002":{
                "name":"hsbuxs",
                "gen":"M",
                "hord":"D",
                "marks1":['35','37','39','33','38','30'],
                "marks2":['30','40','40','40','40','40'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },

            
            "CB22002":{
                "name":"hsbuxs",
                "gen":"F",
                "hord":"D",
                "marks1":['35','37','39','33','38','30'],
                "marks2":['30','40','40','40','40','40'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
            "CB22003":{
                "name":"jksbxzbzs",
                "gen":"M",
                "hord":"H",
                "marks1":['90','90','90','90','90','90'],
                "marks2":['90','90','90','90','90','90'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
            "CB22004":{
                "name":"qwe",
                "gen":"F",
                "hord":"H",
                "marks1":['30','30','30','30','30','30'],
                "marks2":['30','30','30','30','30','30'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
           
            "CB22006":{
                "name":"hsbuxs",
                "gen":"F",
                "hord":"D",
                "marks1":['95','97','99','93','98','90'],
                "marks2":['90','90','90','90','90','90'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
            "CB22007":{
                "name":"jksbxzbzs",
                "gen":"M",
                "hord":"H",
                "marks1":['30','30','30','30','30','30'],
                "marks2":['20','40','30','20','20','20'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            },
            "CB22008":{
                "name":"qwe",
                "gen":"F",
                "hord":"H",
                "marks1":['90','90','90','90','90','90'],
                "marks2":['80','90','90','90','90','90'],
                "marks3":['0','0','0','0','0','0'],
                "marks4":['0','0','0','0','0','0']
            }


    },
    "secyr":{
        "CB21040":{
            "name":"fdd",
            "gen":"M",
            "hord":"D",
            "marks1":['70','80','90','55','60','100'],
            "marks2":['0','0','0','0','0','0'],
            "marks3":['0','0','0','0','0','0'],
            "marks4":['0','0','0','0','0','0']
        }

    },
    "thirdyr":{
        "CB21041":{
            "name":"dfvg",
            "gen":"M",
            "hord":"D",
            "marks1":['90','80','90','95','80','100'],
            "marks2":['80','90','100','70','80','90'],
            "marks3":['90','100','85','99','90','90'],
            "marks4":['0','0','0','0','0','0']

        }

    }
}
predict_marks_and_plot_pass_fail_gender_hostel(actual_dataset)



