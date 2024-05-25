import json
import random


def read_json(path):
    with open(path, "r") as fin:
        data = json.load(fin)
    return data



file_to_merge = [
    'data/VTG-IT/dense_video_caption/ActivityNet_Captions/time_token.json',
    'data/VTG-IT/dense_video_caption/COIN/time_token.json',
    'data/VTG-IT/dense_video_caption/HiREST_step/time_token.json',
    'data/VTG-IT/dense_video_caption/ViTT/time_token.json',
    'data/VTG-IT/dense_video_caption/VTG-IT-DVC/time_token.json',
    'data/VTG-IT/moment_retrieval/DideMo/time_token.json',
    'data/VTG-IT/moment_retrieval/HiREST_grounding/time_token.json',
    'data/VTG-IT/moment_retrieval/QueryD/time_token.json',
    'data/VTG-IT/moment_retrieval/VTG-IT-MR/time_token.json',
    'data/VTG-IT/video_highlight_detection/VTG-IT-VHD/time_token.json',
    'data/VTG-IT/video_summarization/SumMe/time_token.json',
    'data/VTG-IT/video_summarization/TVSum/time_token.json',
    'data/VTG-IT/video_summarization/VTG-IT-VS/time_token.json'
]


merge_data = []

for fi, fpath in enumerate(file_to_merge):
    data = read_json(fpath)
    for i, jterm in enumerate(data):
        data[i]["source"] = file_to_merge[fi].split("/")[-2]
    merge_data.extend(data)
    
random.shuffle(merge_data)

print(len(merge_data))

out_path = "data/VTG-IT/vtg-it/vtg-it-{}k.json".format(round(len(merge_data)/1000), 1)
print("save merge data at {}".format(out_path))
with open(out_path, "w") as fout:
    json.dump(merge_data, fout)
