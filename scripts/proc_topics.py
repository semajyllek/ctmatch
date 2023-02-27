from ctproc.utils import (
  get_sentences, concatenate_data, filter_stops, remove_top_words, get_ents
)
from collections import defaultdict
from zipfile import ZipFile
from pprint import pprint
from pathlib import Path
from lxml import etree
import json
import os
import re



def process_topics(topic_dir_path: Path):
  topics = []
  for _, _, json_files in os.walk(topic_dir_path):
    for i, jf in enumerate(json_files):
      with open(topic_dir_path.joinpath(jf), 'r') as f:
        result = json.load(f)
        result['id'] = result['number']
        del result['number']
        result['age'] = age_to_num_topic(float(result['age']['quantity']),  result['age']['units'])
        topics.append(result)
  return topics




def ag_filter_by_topic(topic_dicts, docs):
  elig_docs_by_topic = defaultdict(set)
  for topic in topic_dicts:
    for doc in docs:
      if (topic['age'] >= doc['eligibility/minimum_age']) and (topic['age'] <= doc['eligibility/maximum_age']):
        if (topic['gender'].lower() == doc['eligibility/gender'].lower()) or (doc['eligibility/gender'].lower() in {'both', "all"}):
          if not re.fullmatch(r'.*[nN]o eligibility criteria.*', doc['eligibility/criteria/textblock']['raw_text']):
            if len(doc['eligibility/criteria/textblock']['raw_text']) > 3:
              elig_docs_by_topic[topic['id']].add(doc['id'])
  return elig_docs_by_topic




def proc_test_topics(xml_filereader):
  text = xml_filereader.read()
  new_topics = []
  for text_block in [t.strip() for t in re.split('<TOP>',text)]:
    topic_dict = {}
    if len(text_block) > 2:
      m = re.search(r'.*<NUM>(?P<id>\d+)<\/NUM>\n\s+<TITLE>(?P<raw_text>.*)\n\s+<\/TOP>', text_block)
      topic_dict['id'] = m.group('id')
      topic_dict['raw_text'] = m.group('raw_text')
      topic_dict['sents'] = get_sentences(m.group('raw_text'))
      topic_dict['ents'] = get_ents(topic_dict['sents'], top_N=2)
      first_sent = topic_dict['sents'][0]
      m = re.search(r'(?P<age_val>\d+)(([- ](?P<age_unit>[^-]+)[- ]old)| ?y\.?o\.?).*(?P<gender>woman| man|female| male|boy|girl) .*', first_sent)
      if m is not None:
        
        topic_dict['age'] = age_to_num_topic(float(m.group('age_val')), m.group('age_unit'))
        topic_dict['gender'] = map_to_gender_yuck(m.group('gender'))
      else:
        topic_dict['age'] = 999.
        topic_dict['gender'] = "Both"
      new_topics.append(topic_dict)
  return new_topics


def transform_topics(
  topics, 
  ignore_fields=[], 
  grab_only_fields=['raw_text'], 
  lower=False, 
  top_words=None, 
  no_stops=False
):
  for topic in topics:
    concat_ents = concatenate_data(
      topic, 
      ignore_fields=ignore_fields, 
      grab_only_fields=grab_only_fields
    ) 
  
    if top_words is not None:
      concat_ents['contents'] = remove_top_words(concat_ents['contents'], top_words)
    
    if no_stops:
      concat_ents['contents'] = filter_stops(concat_ents['contents'])

    topic["contents"] = concat_ents['contents'] if not lower else concat_ents['contents'].lower()
  return topics




