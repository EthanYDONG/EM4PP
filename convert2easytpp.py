import os
import pickle
import re

# 从文件名中提取信息
def extract_info_from_filename(file_name):
    pattern = r"SeqsN(\d+)_Dim(\d+)_K(\d+)_mudelta(\d+\.\d+)_(\w+)\.pkl"
    match = re.match(pattern, file_name)
    global seq_num,k_val, mudelta
    if match:
        seq_num, dim_process, k_val, mudelta, data_type = match.groups()
        dim_process = int(dim_process)
        return dim_process, data_type
    else:
        print("Error: Filename format doesn't match.")
        return None, None

# 加载数据
def load_data_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# 转换数据
def convert_data(original_data, dim_process, data_type):
    try:
        data = []
        for sequence in original_data:
            seq_data = []
            for i in range(len(sequence['Time'])):
                if i == 0:
                    time_since_last_event = 0
                    time_since_start = 0
                else:
                    time_since_last_event = sequence['Time'][i] - sequence['Time'][i - 1]
                    time_since_start = sequence['Time'][i] - sequence['Time'][0]
                type_event = sequence['Mark'][i]

                event_data = {
                    'time_since_last_event': time_since_last_event,
                    'time_since_start': time_since_start,
                    'type_event': type_event
                }
                seq_data.append(event_data)

            seq_dict = {
                'seq_data': seq_data,
                'start_timestamp_seconds': sequence['Time'][0],
                'seq_type': sequence['seq_type']
            }
            data.append(seq_dict)

        converted_data = {
            data_type: data,
            'dim_process': dim_process
        }

        return converted_data
    except Exception as e:
        print(f"Error converting data: {e}")

# 提取文件名信息
folder_path = "/home/wangqingmei/kdd24TPPre/hkstools/hksdata0418delta01/"
output_folder_base = "/home/wangqingmei/kdd24TPPre/hkstools/converted_data_delta0.1/"

file_names = os.listdir(folder_path)
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".pkl"):
        dim_process, data_type = extract_info_from_filename(file_name)
        original_data = load_data_from_pickle(file_path)
        if original_data:
            converted_data = convert_data(original_data, dim_process, data_type)
            if converted_data:
                
                output_folder_name = f"SeqsN{seq_num}_Dim{dim_process}_K{k_val}_mudelta{mudelta}"
                output_folder = os.path.join(output_folder_base, output_folder_name)
                os.makedirs(output_folder, exist_ok=True)
                output_file_path = os.path.join(output_folder, f"{data_type}.pkl")
                with open(output_file_path, 'wb') as f:
                    pickle.dump(converted_data, f)

print("Conversion and saving completed.")
