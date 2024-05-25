import os
import re
import json
from decord import VideoReader
import random

prompts = [
    "Localize the visual content described by the given textual query '<QUERY>' in the video, and output the start and end timestamps in seconds.",
    "Detect and report the start and end timestamps of the video segment that semantically matches the given textual query '<QUERY>'.",
    "Give you a textual query: '<QUERY>' When does the described content occur in the video? Please return the timestamp in seconds.",
    "Locate and describe the visual content mentioned in the text query '<QUERY>' within the video, including timestamps.",
    "The given natural language query '<QUERY>' is semantically aligned with a video moment, please give the start time and end time of the video moment.",
    "Find the video segment that corresponds to the given textual query '<QUERY>' and determine its start and end seconds.",
    "Give you a textual query: '<QUERY>' When does the described content occur in the video?",
    "Identify the video portion that relates to the provided text query '<QUERY>' and provide the starting and ending timestamps in seconds.",
    "Using the textual query '<QUERY>', locate the relevant video segment and mention the start and end times in seconds.",
    "Pinpoint the video section that corresponds to the given text query '<QUERY>' and specify its starting and ending timestamps.",
    "Based on the text query '<QUERY>', determine the appropriate video moment and share the start and end times in seconds.",
    "Match the given textual query '<QUERY>' to a specific part of the video and provide the start and end timestamps.",
    "Detect the video portion that aligns with the provided text query '<QUERY>' and return its starting and ending time in seconds.",
    "Using the text query '<QUERY>', find the relevant part of the video and indicate its start and end times.",
]

Examples = [
    'A specific example is: "45 - 60 seconds, tying shoelaces on a pair of running shoes."',
    'Example: "15 - 25 seconds, flipping through a magazine to find a specific article."',
    'For instance: "120 - 150 seconds, inflating a bicycle tire with a hand pump."',
    'A specific example is: "30 - 45 seconds, watering a small houseplant with a watering can."',
    'A specific example is: "75 - 90 seconds, folding a stack of freshly laundered clothes."',
    'A specific example is: "50 - 65 seconds, writing a short grocery list on a notepad."',
    'Example: "10 - 20 seconds, changing the TV channel using a remote control."',
    'Example: "180 - 210 seconds, brushing teeth with toothpaste and rinsing."',
    'Example: "25 - 35 seconds, plugging in a phone charger and connecting it to a phone."',
    'For instance: "40 - 55 seconds, setting up a board game for a group of friends."',
    'For instance: "70 - 80 seconds, adjusting the height of an office chair for comfort."',
    'For instance: "100 - 120 seconds, searching for a specific book on a bookshelf."',
    'A specific example is: "5 - 15 seconds, applying a postage stamp to an envelope."',
    'Example: "60 - 70 seconds, filling a pet\'s food and water bowls."',
    'For instance: "20 - 30 seconds, replacing a burned-out light bulb in a lamp."'
]

def time_to_seconds(time_str):
    time_parts = time_str.split(':')
    seconds = 0
    if len(time_parts) == 3:
        h, m, s = map(int, time_parts)
        seconds = h * 3600 + m * 60 + s
    elif len(time_parts) == 2:
        m, s = map(int, time_parts)
        seconds = m * 60 + s
    else:
        s = int(time_parts[0])
        seconds = s
    return str(seconds)

def time_to_seconds_B(time_str, event_data=None):
    pattern = re.compile(r'(\d+)\s*(minute|second)s?')
    matches = pattern.findall(time_str)
    seconds = 0
    for value, unit in matches:
        value = int(value)
        if unit == 'minute':
            seconds += value * 60
        elif unit == 'second':
            seconds += value
    return str(seconds)

annotation_path = 'data/yttemporal/raw_annotation.txt'
output_path = 'data/ours/'

data = []

pattern = re.compile(r'\\"events\\": (.*?)', re.DOTALL)
parse_fail_num = 0

with open(annotation_path, 'r') as f:
    for line in f:
        dvc_data_item = {}
        item_data = line
        item_data = item_data.replace('\\n', '\n').replace('\\', '').replace('\"', '"').replace('\n', '').replace('      ', '')

        video_path = re.findall(r'"path": "(.*?)", "prompt"', item_data)[0]
        dvc_data_item['video'] = video_path
        dvc_data_item['QA'] = []
        
        # locate the events from the response
        event_data = re.findall(r'"events":\s*\[(.*?)\]', item_data)
        if len(event_data) < 3:
            parse_fail_num += 1
            continue
        event_data = event_data[2].strip()
        if len(event_data) == 0:
            parse_fail_num += 1
            continue

        ori_event_data = event_data

        # change the time inverval with 0-end to correct seconds
        pattern = re.compile(r'\d+\s*-\s*end')
        times = pattern.findall(event_data)
        if len(times) > 0:
            vr = VideoReader(uri=video_path)
            duration = len(vr) / vr.get_avg_fps()

            # print(f'Video duration: {duration} seconds')
        for time in times:
            event_data = event_data.replace(time, time.replace('end', f'{duration:.1f} seconds'))

        # change the time with format 0 minutes and 0 seconds to seconds
        pattern = re.compile(r'\d+\s*minutes?\s*and\s*\d+\s*seconds?')
        times = pattern.findall(event_data)
        for time in times:
            event_data = event_data.replace(time, time_to_seconds_B(time, ori_event_data))

        # change the time with format 0 minutes 0 seconds to seconds
        pattern = re.compile(r'\d+\s*minutes?\s*\d+\s*seconds?')
        times = pattern.findall(event_data)
        for time in times:
            event_data = event_data.replace(time, time_to_seconds_B(time, ori_event_data))

        # change the time with format 0 minutes
        pattern = re.compile(r'\d+\s*minutes?')
        times = pattern.findall(event_data)
        for time in times:
            event_data = event_data.replace(time, time_to_seconds_B(time, ori_event_data))

        # change the time with format 00:00:00 minutes to seconds
        pattern = re.compile(r'(\d{1,2}(?::\d{2}){0,2})\s*minutes?')
        times = pattern.findall(event_data)
        for time in times:
            event_data = event_data.replace(time + ' minutes', time_to_seconds(time))
            event_data = event_data.replace(time + ' minute', time_to_seconds(time))

        # change the time with format 00:00:00 to seconds
        pattern = re.compile(r'\d{1,2}(?::\d{2}){0,2}')
        times = pattern.findall(event_data)
        for time in times:
            event_data = event_data.replace(time, time_to_seconds(time))

        # # parse the json format event
        pattern = re.compile(r'{\s*"Start\s*-\s*End\s*(seconds?)?\s*":\s*"(\d+\s*-\s*\d+)\s*(seconds?)?\s*",\s*"\s*(Event|event\s+description)\s*":\s*"([^{}]*?)"\s*}')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            for idx, match in enumerate(matches):
                answer = f'{match[1]} seconds, {match[-1]}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                chosen_prompt = prompts[random.randint(0, len(prompts) - 1)]
                chosen_example = '' if random.randint(0, 1) > 0 else ' ' + Examples[random.randint(0, len(Examples) - 1)]
                item_prompt = f'{chosen_prompt} {chosen_example}'.replace('<QUERY>', match[-1])
                dvc_data_item['QA'].append({
                    'q': item_prompt,
                    'a': answer.strip()
                })
            data.append(dvc_data_item)
            continue
        
        # parse 0-0, description

        pattern = re.compile(r'(\d+[\.\d]*\s*-\s*\d+[\.\d]*)\s*(seconds?)?\s*([,:-])?\s*([^{}"]*?)"')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            for match in matches:
                answer = f'{match[0]} seconds, {match[-1]}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                chosen_prompt = prompts[random.randint(0, len(prompts) - 1)]
                item_prompt = f'{chosen_prompt}'.replace('<QUERY>', match[-1])
                dvc_data_item['QA'].append({
                    'q': item_prompt,
                    'a': answer.strip()
                })
        data.append(dvc_data_item)

    print(f'{len(data)} data parsed, {parse_fail_num} parse failed')

    out_file_path = output_path + f'tvg_{len(data) // 1000}k_v3.json'
    with open(out_file_path, 'w') as f:
        json.dump(data, f)