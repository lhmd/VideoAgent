import ast
import copy
import json
import os
import pickle
import random
import re
import socket
import subprocess
import sys
from io import StringIO
from multiprocessing import Process

import pandas as pd
import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import AzureChatOpenAI
from omegaconf import OmegaConf

from captioning import Captioning
from encoder import load_xcomposer
from reid import ReID
from segment_feature import SegmentFeature
from tools import ToolKit
from tracking import Tracking
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://api.tonggpt.mybigai.ac.cn/proxy/eastus"
os.environ["AZURE_OPENAI_API_KEY"] = "b47bf7405b388f3dcb00aa452a2aefdf"


def ReActAgent(video_path, question, base_dir='preprocess', vqa_tool='videollava', vqa_asset=None, frame_based=False, use_reid=True, openai_api_key='your_openai_api_key', prompt_path='prompts/prompt.txt'):
    assert vqa_tool in ['videollava', 'gpt-4v', 'xcomposer']
    # __import__('ipdb').set_trace()
    toolkit = ToolKit(video_path=video_path, base_dir=base_dir, vqa_tool=vqa_tool,
                      vqa_asset=vqa_asset, frame_based=frame_based, use_reid=use_reid, openai_api_key=openai_api_key)
    @tool
    def object_memory_querying(question):
        """Given a question about open-vocabulary objects such as 'how many people are there in the video?' or 'In which segments did the brown dog appear?', this tool will give the answer based on the object memory."""
        print("########Tool object_memory_querying########")
        @tool
        def database_querying(program):
            """given a MySQL program, this tool will query the database and return the results."""
            ans = toolkit.query_database(program=program)
            return '\n'+ans+'\n'
        @tool
        def open_vocabulary_object_retrieval(description):
            """given an open-vocabulary description of an object or a person (frying pan, person in red clothes e.g.), this tool will return the possible candidate object IDs that satisfy the description."""
            ans = toolkit.retrieve_candidate_objects(description=description)
            return '\n'+ans+'\n'
        prompt = hub.pull("hwchase17/react")
        with open('prompts/database_query_prompt.txt') as f:
            t = f.read()
        prompt.template = t
        llm = AzureChatOpenAI(temperature=0, openai_api_version='2024-02-01', azure_deployment="gpt-4o-2024-05-13", streaming=False)
        tools = [database_querying, open_vocabulary_object_retrieval]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        print("Step into agent_executor in object_memory_querying")
        original_stdout = sys.stdout
        output_catcher = StringIO()
        sys.stdout = output_catcher
        agent_executor.invoke({"input": question})
        sys.stdout = original_stdout
        output_catcher.seek(0)
        lines = output_catcher.readlines()
        color_pattern = re.compile(r'\x1B\[[0-9;]*[mK]')
        answer = None
        for line in lines:
            print(line)
            if line.startswith("Final Answer: "):
                line = color_pattern.sub('', line)
                line = line.replace("Final Answer: ", "")
                answer = line
        return answer

    @tool
    def segment_localization(description):
        """Given a textual description, this tool will return the top-5 candidate segments that are most relevant to the description."""
        print("########Tool segment_localization########")
        answer = toolkit.segment_localization(description, k=5)
        return '\n'+answer+'\n'

    @tool
    def caption_retrieval(input_tuple):
        """given an input tuple (start_segment_ID, end_segment_ID), this tool will retrieve all the captions between the two segments, 15 captions at most. end_segment_ID < start_segment_ID + 15."""
        print("########Tool caption_retrieval########")
        input_tuple = ast.literal_eval(input_tuple)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        answer = toolkit.caption_retrieval(int(input_tuple[0]), int(input_tuple[1]))
        return '\n'+answer+'\n'

    @tool
    def visual_question_answering(input_tuple):
        """Given an input tuple (question, segment_ID), this tool will focus on the video segments starting from segment_ID-1 to segment_ID+1. It will return the description of the video segment and the answer to the question based on the segment."""
        print("########Tool visual_question_answering########")
        input_tuple = ast.literal_eval(input_tuple)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        question = input_tuple[0]
        segment_id = int(input_tuple[1])
        answer = toolkit.visual_question_answering(question, segment_id)
        return '\n'+answer+'\n'

    prompt = hub.pull("hwchase17/react")
    with open(prompt_path) as f:
        t = f.read()

    prompt.template = t
    #print(prompt)
    llm = AzureChatOpenAI(temperature=0, openai_api_version='2024-02-01', azure_deployment="gpt-4o-2024-05-13", streaming=False)
    tools = [caption_retrieval, segment_localization, visual_question_answering, object_memory_querying]
    agent = create_react_agent(llm, tools, prompt)
    print(question)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    print("Step into agent_executor")
    original_stdout = sys.stdout
    output_catcher = StringIO()
    sys.stdout = output_catcher
    agent_executor.invoke({"input": question})
    sys.stdout = original_stdout

    output_catcher.seek(0)
    lines = output_catcher.readlines()
    # print("lines: ", lines)
    color_pattern = re.compile(r'\x1B\[[0-9;]*[mK]')
    answer = None
    log = ""
    for line in lines:
        print(line)
        new_line = color_pattern.sub('', line)
        log += new_line
        if new_line.startswith("Final Answer: "):
            answer = new_line.replace("Final Answer: ", "")
    return answer, log


def preprocess(video_path_list, base_dir='preprocess', frame_based=False, show_tracking=False):
    required_files = [
        "segment2id.json",
        "captions_xcomposer.json" if frame_based else "captions_lavila.json",
        "segment_textual_embedding_xcomposer.pkl" if frame_based else "segment_textual_embedding_lavila.pkl",
        "segment_visual_embedding_frame.pkl" if frame_based else "segment_visual_embedding_video.pkl",
        "tracking.pkl",
        "reid.pkl",
        "tid2clip.pkl",
        "tid2dinov2.pkl",
        "uid2clip.pkl",
        "reid.mp4"
        ]
    def check_has_been_preprocessed(video_path, required_files):
        base_name = os.path.basename(video_path).replace(".mp4", "")
        video_dir = os.path.join(base_dir, base_name)
        if not os.path.exists(video_dir):
            return False
        files = os.listdir(video_dir)
        for f in required_files:
            if f not in files:
                return False
        return True

    # step 1: captioning
    preprocess_list = []
    for video_path in video_path_list:
        if not check_has_been_preprocessed(video_path, required_files[:2]):
            preprocess_list.append(video_path)
    print('A total of {} videos need to be captioned.'.format(len(set(preprocess_list))))
    if preprocess_list:
        captioning = Captioning(video_path_list=preprocess_list,
                                base_dir=base_dir, frame_based=frame_based)
        captioning.run()

    # step 2: segment feature
    preprocess_list = []
    for video_path in video_path_list:
        if not check_has_been_preprocessed(video_path, required_files[2:4]):
            preprocess_list.append(video_path)
    print('A total of {} videos need to be segmented.'.format(len(set(preprocess_list))))
    if preprocess_list:
        temporal_feature = SegmentFeature(video_path_list=preprocess_list,
                                        base_dir=base_dir, frame_based=frame_based)
        temporal_feature.run()

    # step 3: build object memory
    preprocess_list = []
    for video_path in video_path_list:
        if not check_has_been_preprocessed(video_path, required_files[4:]):
            preprocess_list.append(video_path)
    print('A total of {} videos need to be tracked & re-IDed.'.format(len(set(preprocess_list))))
    if preprocess_list:
        tracking = Tracking(video_path_list=preprocess_list,
                            base_dir=base_dir,
                            tracking_fps=15,
                            sample_num=5,
                            show=show_tracking)
        tracking.run()
        reid = ReID(video_path_list=preprocess_list,
                    base_dir=base_dir)
        reid.run()


def eval_openeqa():
    config = OmegaConf.load('config/default.yaml')
    openai_api_key = config['openai_api_key']
    use_reid = config['use_reid']
    vqa_tool = config['vqa_tool']
    base_dir = config['base_dir']

    vqa_tool = 'xcomposer'
    frame_based = True

    data = json.load(open('/mnt/bigai_ml/all_datasets/OpenEQA/open-eqa-v0.json', 'r'))
    video_question_list = [i['question'] for i in data]
    video_qid_list = [i['question_id'] for i in data]
    video_path_list = []
    for i in data:
        id = i['episode_history']
        eid = i['episode_history'].split('/')[-1]
        video_path_list.append(f'/mnt/bigai_ml/all_datasets/OpenEQA/frames/{id}/{eid}.mp4')
    video_answer_list = [i['answer'] for i in data]

    random.seed(12345)
    zipped_lists = list(zip(video_question_list, video_qid_list, video_path_list, video_answer_list))
    random.shuffle(zipped_lists)
    video_question_list, video_qid_list, video_path_list, video_answer_list = zip(*zipped_lists)
    #sample_size = int(len(video_question_list) * 0.1)
    sample_size = 10
    video_question_list = list(video_question_list)[:sample_size]
    video_qid_list = list(video_qid_list)[:sample_size]
    video_path_list = list(video_path_list)[:sample_size]
    video_answer_list = list(video_answer_list)[:sample_size]

    chains = main(video_path_list=video_path_list,
            video_question_list=video_question_list,
            video_answer_list=video_answer_list,
            base_dir=base_dir,
            vqa_tool=vqa_tool,
            frame_based=frame_based,
            use_reid=use_reid,
            openai_api_key=openai_api_key)

    results = []
    for i in zip(video_question_list, video_qid_list, chains):
        results.append({
            'question': i[0],
            'question_id': i[1],
            'answer': i[2][0],
            'chain': i[2][1],
        })

    with open('openeqa_eval_results_0.1_all.json', 'w') as f:
        json.dump(results, f)

    result = subprocess.run(['python', 'eval_openeqa.py', 'openeqa_eval_results_0.1_all.json', '--dataset', '/mnt/bigai_ml/all_datasets/OpenEQA/open-eqa-v0.json'], capture_output=True, text=True)

    # Print the stdout of the bash script
    print(result.stdout)


def eval_videomme():
    config = OmegaConf.load('config/default.yaml')
    openai_api_key = config['openai_api_key']
    use_reid = config['use_reid']
    vqa_tool = config['vqa_tool']
    base_dir = 'preprocess_videomme'

    vqa_tool = 'xcomposer' # or videollava
    frame_based = True

    data = pd.read_parquet('/mnt/bigai_ml/all_datasets/Video-MME/videomme/test-00000-of-00001.parquet')
    results = {}
    for i in data.iloc:
        if i['duration'] == 'long':
            continue
        if i['video_id'] not in results:
            results[i['video_id']] = {
            'video_url': i['videoID'],
            'duration_category': i['duration'],
            'video_category': i['domain'],
            'video_subcategory': i['sub_category'],
            'questions': []
        }
        results[i['video_id']]["questions"].append({
            'question_id': i['question_id'],
            'task_type': i['task_type'],
            'question': i['question'],
            'choices': list(i['options']),
            'answer': i['answer'],
            'response': '',
        })

    random.seed(12345)
    selected_videos = list(results.keys())
    random.shuffle(selected_videos)
    sample_size = int(len(selected_videos) * 0.002)
    selected_videos = selected_videos[:sample_size]

    video_task_list = []
    video_path_list = []
    video_question_list = []
    video_answer_list = []
    for video_id in selected_videos:
        for qid, i in enumerate(results[video_id]['questions']):
            question = i['question'] + ' \n ' + ' \n '.join(i['choices'])
            video_task_list.append((video_id, qid+1))
            video_question_list.append(question)
            video_answer_list.append(i['answer'])
            video_path_list.append(f'/mnt/bigai_ml/all_datasets/Video-MME/video/{results[video_id]["video_url"]}.mp4')

    chains = main(video_path_list=video_path_list,
            video_question_list=video_question_list,
            video_answer_list=video_answer_list,
            base_dir=base_dir,
            vqa_tool=vqa_tool,
            frame_based=frame_based,
            use_reid=use_reid,
            openai_api_key=openai_api_key,
            prompt_path='prompts/multiple_choice_prompt_videomme.txt')

    dump = []
    for i in zip(video_question_list, video_task_list, chains):
        assert results[i[1][0]]['questions'][i[1][1]-1]['question_id'] == '-'.join([str(j) for j in i[1]])
        results[i[1][0]]['questions'][i[1][1]-1]['response'] = i[2][0]
        dump.append({
            'question': i[0],
            'answer': i[2][0],
            'chain': i[2][1],
        })

    with open('videomme_results_0.1_all.json', 'w') as f:
        json.dump(dump, f)
    final_result = []
    for k, v in results.items():
        tmp = copy.deepcopy(v)
        tmp['video_id'] = k
        final_result.append(tmp)
    with open('videomme_eval_results_0.1_all.json', 'w') as f:
        json.dump(final_result, f)

    result = subprocess.run(['python', 'eval_videomme.py', '--video_duration_type', 'short,medium',  '--results_file', 'videomme_eval_results_0.1_all.json'], capture_output=True, text=True)

    # Print the stdout of the bash script
    print(result.stdout)

def eval_generalqa():
    config = OmegaConf.load('config/default.yaml')
    openai_api_key = config['openai_api_key']
    use_reid = config['use_reid']
    vqa_tool = config['vqa_tool']
    base_dir = config['base_dir']
    frame_based = False

    video_path_list = [
        "sample_videos/boats.mp4",
        "sample_videos/talking.mp4",
        "sample_videos/books.mp4",
        "sample_videos/painting.mp4",
        "sample_videos/kitchen.mp4"
    ]
    video_question_list = [
        "How many boats are there in the video?",
        "From what clue do you know that the woman with black spectacles at the start of the video is married?",
        "Based on the actions observed, what could be a possible motivation or goal for what c is doing in the video?",
        "What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?",
        "Is there a microwave in the kitchen?"
    ]
    chains = main(video_path_list=video_path_list,
         video_question_list=video_question_list,
          base_dir=base_dir,
          vqa_tool=vqa_tool,
          frame_based=frame_based,
          use_reid=use_reid,
          openai_api_key=openai_api_key)


def main(video_path_list, video_question_list, video_answer_list, base_dir='preprocess', vqa_tool='videollava', frame_based=False, use_reid=True, openai_api_key='sk-proj-f7tYXKiSkl6f6z35upQJT3BlbkFJmGpl5Kvu6xqpEG0H7niu', prompt_path='prompts/prompt.txt'):
    
    # TEST
    # random.shuffle(video_path_list)
    video_path_list = video_path_list[:3]
    
    preprocess(video_path_list=video_path_list,
               base_dir=base_dir,
               show_tracking=False,
               frame_based=frame_based)

    if frame_based:
        assert vqa_tool == 'xcomposer'
        vqa_model, vqa_tokenizer = load_xcomposer()
        vqa_asset = (vqa_model, vqa_tokenizer)
    else:
        vqa_asset = None

    question_num = len(video_question_list)
    
    # TEST
    question_num = 3
    print("#################")
    print("question_num: ", question_num)
    print("#################")
    
    chains = []
    for i in tqdm.tqdm(range(question_num)):
        try:
            ret = ReActAgent(video_path=video_path_list[i], question=video_question_list[i], base_dir=base_dir, vqa_tool=vqa_tool, vqa_asset=vqa_asset, frame_based=frame_based, use_reid=use_reid, openai_api_key=openai_api_key, prompt_path=prompt_path)
            print("ret: ", ret)
            chains.append(ret)
            print(f'Q: {video_question_list[i]}; GT: {video_answer_list[i]}')
        except:
            print("Error in processing the question.")
            ret = ''
    return chains


if __name__ == '__main__':
    # eval_generalqa()
    eval_openeqa()
    # eval_videomme()
