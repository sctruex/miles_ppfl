import numpy as np
import pandas as pd
import os


def find_user_id(file_path=''):
    # user ids are found at index=0 of every line
    user_ids = []
    X = np.genfromtxt(file_path, dtype=str, delimiter=',', skip_header=1)
    for user in X:
        user_ids.append(user[0])
    return user_ids


def create_annotate_df(user_id, file_path):
    df = pd.read_csv(f"{file_path}/{user_id}_annotate.csv")
    annotate_df = pd.DataFrame(columns=['Start', 'Stop', 'Action'])
    for i in range(0, len(df), 2):
        annotate_df.loc[i, 'Start'] = df.loc[i, 'unix time']
        annotate_df.loc[i, 'Stop'] = df.loc[i+1, 'unix time']
        annotate_df.loc[i, 'Action'] = df.loc[i, 'Activity']
    annotate_df.index = [i for i in range(len(annotate_df))]
    return annotate_df


def create_sensor_df(user_id, file_path):
    df = pd.read_csv(f"{file_path}/{user_id}_sensorData.csv")
    df = df[df['Sensor'] != "LTR-506ALS Light sensor"]
    df.index = [i for i in range(len(df))]
    columns = ['Start', 'Stop', 'MPL Magnetic Field X', 'MPL Magnetic Field Y', 'MPL Magnetic Field Z',
               'MPL Rotation Vector X', 'MPL Rotation Vector Y', 'MPL Rotation Vector Z', 'MPL Linear Acceleration X',
               'MPL Linear Acceleration Y', 'MPL Linear Acceleration Z', 'MPL Gravity X', 'MPL Gravity Y',
               'MPL Gravity Z', 'MPL Gyroscope X', 'MPL Gyroscope Y', 'MPL Gyroscope Z', 'MPL Accelerometer X',
               'MPL Accelerometer Y', 'MPL Accelerometer Z', 'Action']
    sensors = {'MPL Magnetic Field': ['MPL Magnetic Field X', 'MPL Magnetic Field Y', 'MPL Magnetic Field Z'],
               'MPL Rotation Vector': ['MPL Rotation Vector X', 'MPL Rotation Vector Y', 'MPL Rotation Vector Z'],
               'MPL Linear Acceleration': ['MPL Linear Acceleration X', 'MPL Linear Acceleration Y', 'MPL Linear Acceleration Z'],
               'MPL Gravity': ['MPL Gravity X', 'MPL Gravity Y', 'MPL Gravity Z'],
               'MPL Gyroscope': ['MPL Gyroscope X', 'MPL Gyroscope Y', 'MPL Gyroscope Z'],
               'MPL Accelerometer': ['MPL Accelerometer X', 'MPL Accelerometer Y', 'MPL Accelerometer Z']}
    sensor_lst = ['MPL Magnetic Field', 'MPL Rotation Vector',
                  'MPL Linear Acceleration', 'MPL Gravity', 'MPL Gyroscope', 'MPL Accelerometer']
    sensor_lst.sort()
    sensor_df = pd.DataFrame(columns=columns)
    index = 0
    while (index+6) <= len(df):
        temp_df = df[index:index+6]
        temp_df.index = [i for i in range(len(temp_df))]
        lst = []
        for j in range(len(temp_df)):
            lst.append(temp_df.loc[j, 'Sensor'])
        lst.sort()
        if sensor_lst != lst:
            index += 1
        else:
            sensor_df.loc[index, 'Start'] = temp_df['Unix Time'].min()
            sensor_df.loc[index, 'Stop'] = temp_df['Unix Time'].max()
            for val in range(len(temp_df)):
                sensor = sensor_lst[val]
                sensor_df.loc[index, sensors[sensor]
                              [0]] = temp_df.loc[val, "Value1"]
                sensor_df.loc[index, sensors[sensor]
                              [1]] = temp_df.loc[val, "Value2"]
                sensor_df.loc[index, sensors[sensor]
                              [2]] = temp_df.loc[val, "Value3"]
            index += 6
    sensor_df.index = [i for i in range(len(sensor_df))]
    return sensor_df


def add_action(user_id, file_path=''):
    annotate_df = create_annotate_df(user_id, file_path)
    sensor_df = create_sensor_df(user_id, file_path)
    for i in range(len(annotate_df)):
        start = annotate_df.loc[i, 'Start']
        stop = annotate_df.loc[i, 'Stop']
        temp_sensor_df = sensor_df[((sensor_df['Start'] >= start) &
                                    (sensor_df['Stop'] <= stop))]
        if annotate_df.loc[i, 'Action'] == 'eat':
            index_lst = (temp_sensor_df.index).tolist()
            for j in index_lst:
                sensor_df.loc[j, 'Action'] = 1  # 'eating'
        else:
            for j in temp_sensor_df.index:
                sensor_df.loc[j, 'Action'] = 0  # 'not eating'
    action_df = sensor_df[(sensor_df['Action'] == 1) |
                          (sensor_df['Action'] == 0)]
    return action_df


def stack_data(file_path):
    user_lst = find_user_id(f'{file_path}/Demographics.csv')
    final_df = pd.DataFrame()
    for user in user_lst:
        user_df = pd.read_csv(f'{file_path}/client_data/{user}_data.csv')
        final_df = pd.concat([final_df, user_df]).reset_index(drop=True)
    final_df.to_csv(f'{file_path}/gleam.csv', index=False)


def read_gleam_data(file_path=''):
    user_lst = find_user_id(f'{file_path}/Demographics.csv')
    for user in user_lst:
        print(f"considering {user}")
        # not os.path.isfile(f"./data_utils/csv_datasets/gleam/client_data/{user}_data.csv"):
        if True:
            print(f"csv for {user} does not exist, creating now")
            df = add_action(user, f"{file_path}/client_data")
            df.to_csv(
                f"./data_utils/csv_datasets/gleam/client_data/{user}_data.csv", index=False)
    print(f"creating aggregate file")
    stack_data(file_path)


if __name__ == "__main__":
    read_gleam_data("./data_utils/csv_datasets/gleam/")
