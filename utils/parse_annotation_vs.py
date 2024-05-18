import os
import re
import json
from decord import VideoReader
import random


prompts = [
    "After viewing the assigned video, identify the moments that align with the description: '<query_placeholder>'. Record the timecodes of these key scenes and assess their prominence using a relevance scale.",
    "Examine the given video and pinpoint the segments that correspond to the description: '<query_placeholder>'. Note down the timeframes of these notable instances and rate their conspicuousness using a significance scoring system.",
    "Scrutinize the supplied video and locate the portions that match the description: '<query_placeholder>'. Jot down the timestamps of these remarkable occurrences and estimate their distinctiveness using a visibility ranking.",
    "Study the presented video and detect the sections that coincide with the description: '<query_placeholder>'. Log the timing of these striking events and evaluate their notability using an attention-grabbing metric.",
    "Observe the included video and spot the parts that are in line with the description: '<query_placeholder>'. Document the instances of these highlights and gauge their noticeability using a standout scoring method.",
    "Analyze the featured video and find the episodes that relate to the description: '<query_placeholder>'. Register the time markers of these remarkable moments and measure their prominence using an impact rating system.",
    "Review the provided video and determine the scenes that connect with the description: '<query_placeholder>'. Capture the time points of these significant events and appraise their salience using a prominence evaluation scale.",
    "Explore the specified video and discern the segments that resonate with the description: '<query_placeholder>'. Chronicle the timestamps of these attention-grabbing instances and assess their importance using a distinctiveness index.",
    "Investigate the accompanying video and recognize the sections that correspond with the description: '<query_placeholder>'. Note the time intervals of these standout occurrences and estimate their visibility using a prominence scoring technique."
    "Peruse the given video and identify the moments that harmonize with the description: '<query_placeholder>'. Record the time indications of these exceptional scenes and evaluate their conspicuousness using a saliency rating system.",
    "After viewing the assigned video, identify the moments that align with the description: '<query_placeholder>'. Record the timecodes of these key scenes and assess their prominence using a relevance scale. The output format should be as follows: 'Second of highlight frames, significance score, description'.",
    "Examine the given video and pinpoint the segments that correspond to the description: '<query_placeholder>'. Note down the timeframes of these notable instances and rate their conspicuousness using a significance scoring system. The output format should be: 'Second of highlight frames, significance score, description'.",
    "Scrutinize the supplied video and locate the portions that match the description: '<query_placeholder>'. Jot down the timestamps of these remarkable occurrences and estimate their distinctiveness using a visibility ranking. The output format should be as follows: 'Second of highlight frames, significance score, description'.",
    "Study the presented video and detect the sections that coincide with the description: '<query_placeholder>'. Log the timing of these striking events and evaluate their notability using an attention-grabbing metric. The output format should be: 'Second of highlight frames, significance score, description'.",
    "Observe the included video and spot the parts that are in line with the description: '<query_placeholder>'. Document the instances of these highlights and gauge their noticeability using a standout scoring method. Present the results in the following format: 'Second of highlight frames, significance score, description'.",
    "Analyze the featured video and find the episodes that relate to the description: '<query_placeholder>'. Register the time markers of these remarkable moments and measure their prominence using an impact rating system. Present the results in the following format: 'Second of highlight frames, significance score, description'.",
    "Review the provided video and determine the scenes that connect with the description: '<query_placeholder>'. Capture the time points of these significant events and appraise their salience using a prominence evaluation scale. The output should be presented as: 'Second of highlight frames, significance score, description'.",
    "Explore the specified video and discern the segments that resonate with the description: '<query_placeholder>'. Chronicle the timestamps of these attention-grabbing instances and assess their importance using a distinctiveness index. The output should be presented as: 'Second of highlight frames, significance score, description'.",
    "Investigate the accompanying video and recognize the sections that correspond with the description: '<query_placeholder>'. Note the time intervals of these standout occurrences and estimate their visibility using a prominence scoring technique. The output should include: 'Second of highlight frames, significance score, description'."
    "Peruse the given video and identify the moments that harmonize with the description: '<query_placeholder>'. Record the time indications of these exceptional scenes and evaluate their conspicuousness using a saliency rating system. The output should include: 'Second of highlight frames, significance score, description'."
]

captions = [
    "Crucial moments in the video.",
    "Key scenes captured in the video.",
    "Significant frames from the video.",
    "Essential moments of the video.",
    "Noteworthy frames in the video.",
    "Pivotal points in the video.",
    "Prominent frames throughout the video.",
    "Important visual highlights of the video.",
    "Remarkable frames of the video."
]

Examples = [
    'A specific example is: "45 seconds, significance score: 4.1, tying a double knot on a pair of running shoes".',
    'Example: "120 seconds, significance score: 3.7, boiling water in an electric kettle".',
    'For instance: "30 seconds, significance score: 2.3, flipping a pancake on a hot griddle".',
    'Example: "60 seconds, significance score: 1.9, inserting a USB drive into a laptop".',
    'Example: "75 seconds, significance score: 4.0, peeling an orange with a knife".',
    'Example: "105 seconds, significance score: 3.8, adjusting the rearview mirror in a car".',
    'Example: "50 seconds, significance score: 1.7, opening a can of soda".',
    'Example: "80 seconds, significance score: 4.6, folding a map back into its original form".',
    'Sample: "55 seconds, significance score: 2.5, putting on a pair of glasses".',
    'Example: "100 seconds, significance score: 3.9, setting up a new WiFi router".'
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
output_path = 'yourpath/data/vtg-it/'

data = []

pattern = re.compile(r'\\"highlights\\": (.*?)', re.DOTALL)
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
        event_data = re.findall(r'"highlights":\s*\[(.*?)\]', item_data)
        if len(event_data) < 3:
            parse_fail_num += 1
            continue
        event_data = event_data[2].strip()
        if len(event_data) == 0:
            parse_fail_num += 1
            continue

        # print(event_data)

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

        # print('*' * 50)
        # print(event_data)
        # print(ori_event_data)
        # print('*' * 50)

        caption = captions[random.randint(0, len(captions) - 1)]

        # parse the json format event
        pattern = re.compile(r'{\s*"\s*Second\s*(of\s*highlight\s*frames)?\s*":\s*"\s*(\d+)\s*(seconds?)?\s*"\s*,\s*"\s*(significance\s*score)\s*"\s*:\s*"?\s*(\d+)\s*"?\s*,\s*"\s*description\s*"\s*:\s*"([^{}]*?)"\s*}')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            answer = ''
            for match in matches:
                answer += f'{match[1]} seconds, significance score: {match[4]}, {caption}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                else:
                    answer += ' '
            dvc_data_item['QA'][0]['q'] = item_prompt.replace('<query_placeholder>', caption.strip())
            dvc_data_item['QA'][0]['a'] = answer.strip()
            data.append(dvc_data_item)
            # print(dvc_data_item)
            continue

        # point level annotations
        pattern = re.compile(r'"\s*(\d+)\s*(seconds?)?\s*[:,-]\s*([Ss]ignificance\s*score:)\s*(\d+)\s*,\s*([^"]*?)"')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            answer = ''
            for match in matches:
                answer += f'{match[0]} seconds, significance score: {match[-2]}, {caption}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                else:
                    answer += ' '
        
            dvc_data_item['QA'][0]['q'] = item_prompt.replace('<query_placeholder>', caption.strip())
            dvc_data_item['QA'][0]['a'] = answer.strip()
            data.append(dvc_data_item)
            # if len(matches) < 5:
            #     print('*' * 50)
            #     print(dvc_data_item)
            #     print(event_data)
            #     print('*' * 50)
            continue

        pattern = re.compile(r'"\s*(\d+)\s*(seconds?)?\s*[,]\s*([Ss]ignificance\s*score:)?\s*(\d+)\s*,\s*([^"]*?)"')
        matches = pattern.findall(event_data)
        if len(matches) > 0:
            answer = ''
            for match in matches:
                answer += f'{match[0]} seconds, significance score: {match[-2]}, {caption}'.strip()
                if answer[-1] != '.':
                    answer += '. '
                else:
                    answer += ' '
        
            dvc_data_item['QA'][0]['q'] = item_prompt.replace('<query_placeholder>', caption.strip())
            dvc_data_item['QA'][0]['a'] = answer.strip()
            data.append(dvc_data_item)
            # if len(matches) < 5:
            #     print('*' * 50)
            #     print(dvc_data_item)
            #     print(event_data)
            #     print('*' * 50)
            continue

        # print(event_data)

        
    #     # parse 0-0, description
    #     answer = ''
    #     matched_num = 0

    #     pattern = re.compile(r'(\d+[\.\d]*\s*-\s*\d+[\.\d]*)\s*(seconds?)?\s*([,:-])?\s*([^{}"]*?)"')
    #     matches = pattern.findall(event_data)
    #     if len(matches) > 0:
    #         for match in matches:
    #             answer += f'{match[0]} seconds, {match[-1]}'.strip()
    #             if answer[-1] != '.':
    #                 answer += '. '
    #             else:
    #                 answer += ' '
    #             matched_num += 1
        
    #     dvc_data_item['QA'][0]['a'] = answer.strip()
    #     data.append(dvc_data_item)

    print(f'{len(data)} data parsed, {parse_fail_num} parse failed')

    out_file_path = output_path + f'vs_{len(data) // 1000}k_v2.json'
    with open(out_file_path, 'w') as f:
        json.dump(data, f)