import os
import re
import json
from decord import VideoReader
import random

prompts = [
    'Please locate a series of events in the video, output the start and end timestamps of each event, and describe each event in sentences. The output format of each predicted event should be as follows: "Start - End seconds, event description".',
    'Identify various events in the video and provide their start and end times along with a brief description.',
    'Watch the video and list out different events with their respective start and end timestamps, followed by a short explanation.',
    'Analyze the video and pinpoint multiple events, mentioning their starting and ending times as well as a summary of each event.',
    'Observe the video and detect several occurrences, noting down their beginning and end times accompanied by a concise description.',
    'Examine the video and locate a number of events, stating their commencement and completion timestamps and a succinct explanation.',
    'Review the video and find various happenings, indicating their start and finish times and a brief account of each event.',
    'Inspect the video and recognize multiple activities, mentioning their onset and termination timestamps along with a short description.',
    'Study the video and spot several events, providing their starting and concluding times as well as a summary of each occurrence.',
    'Watch the video carefully and list down distinct events, including their beginning and ending timestamps and a concise explanation.',
    'Evaluate the video and identify a series of actions, stating their initiation and completion times accompanied by a brief description.',
    'Assess the video and discover various incidents, noting their start and end points along with a short account of each event.',
    'Scrutinize the video and determine multiple occurrences, providing their initial and final timestamps as well as a summary of each action.'
]

Examples = [
    'A specific example is: "90 - 102 seconds, spreading butter on two slices of white bread".',
    'Example: "45 - 58 seconds, pouring orange juice into a glass".',
    'For instance: "22 - 35 seconds, tying shoelaces on running shoes".',
    'Example: "267 - 480 seconds, folding a blue T-shirt".',
    'Example: "18 - 32 seconds, watering plants in the garden".',
    'Example: "39 - 52 seconds, slicing an apple into pieces".',
    'Example: "100 - 124 seconds, stirring coffee in a white mug".',
    'Example: "55 - 68 seconds, adjusting a bicycle seat height".',
    'Sample: "29 - 42 seconds, flipping pancakes on a frying pan".',
    'Example: "760 - 789 seconds, vacuuming the living room carpet".',
    'Example: "3 - 16 seconds, opening a laptop and logging in".',
    'Example: "8 - 11 seconds, pouring milk into a bowl of cereal".',
    'Sample: "34 - 47 seconds, walking a dog in the park".'
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

annotation_path = 'yourpath/data/yttemporal/raw_annotation.txt'
output_path = 'yourpath/data/ours/'

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
        dvc_data_item['QA'] = [{}]
        chosen_prompt = prompts[random.randint(0, len(prompts) - 1)]
        chosen_example = '' if random.randint(0, 1) > 0 else ' ' + Examples[random.randint(0, len(Examples) - 1)]
        item_prompt = f'{chosen_prompt}{chosen_example}'
        
        dvc_data_item['QA'][0]['q'] = item_prompt
        
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
            answer = ''
            for match in matches:
                answer += f'{match[1]} seconds, {match[-1]}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                else:
                    answer += ' '
            dvc_data_item['QA'][0]['a'] = answer.strip()
            data.append(dvc_data_item)
            continue
        
        # parse 0-0, description
        answer = ''
        matched_num = 0

        pattern = re.compile(r'(\d+[\.\d]*\s*-\s*\d+[\.\d]*)\s*(seconds?)?\s*([,:-])?\s*([^{}"]*?)"')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            for match in matches:
                answer += f'{match[0]} seconds, {match[-1]}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                else:
                    answer += ' '
                matched_num += 1
        
        dvc_data_item['QA'][0]['a'] = answer.strip()
        data.append(dvc_data_item)

    print(f'{len(data)} data parsed, {parse_fail_num} parse failed')

    out_file_path = output_path + f'dvc_{len(data) // 1000}k.json'
    with open(out_file_path, 'w') as f:
        json.dump(data, f)