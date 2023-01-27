# Process tons of files of the form ClinicalTrials.2021-04-27.part1/NCT0093xxxx/NCT00934219.xml, now on local disk

from drive.MyDrive.trec21_ct_full.trec_ct.scripts.move_negs import move_neg
from drive.MyDrive.trec21_ct_full.trec_ct.scripts.trec21_vis import *
from collections import defaultdict, Counter
from scispacy.linking import EntityLinker
from typing import List, Dict, Union
from matplotlib import pyplot as plt
from negspacy.negation import Negex
from zipfile import ZipFile
from tarfile import TarFile
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import numpy as np
import random
import pprint
import pandas
import spacy
import copy
import glob
import json
import os
import re


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
nlp = spacy.load("en_core_sci_md") 

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
nlp.add_pipe("negex")
linker = nlp.get_pipe("scispacy_linker")

STOP_WORDS = nlp.Defaults.stop_words
DONT_ALIAS = {"yo", "girl", "boy", "er", "changes", "patient", "male", "female", "age"}
REMOVE_WORDS = ["criteria", "include", "exclude", "inclusion", "exclusion", "eligibility"]


#import language_tool_python
#tool = language_tool_python.LanguageTool('en-US')


ct_file_pattern = re.compile("\S+(NCT.*\.xml)")

empty_pattern = re.compile('[\n\s]+')
inc_only_pattern = re.compile('[\s\n]+[Ii]nclusion [Cc]riteria:?([\w\W ]*)')
exc_only_pattern = re.compile('[\n\r ]+[Ee]xclusion [Cc]riteria:?([\w\W ]*)')
age_pattern = re.compile('(?P<age>\d+) *(?P<units>\w+).*')
year_pattern = re.compile('(?P<year>[yY]ears?.*)')
month_pattern = re.compile('(?P<month>[mM]o(?:nth)?)')
week_pattern = re.compile('(?P<week>[wW]eeks?)')

both_inc_and_exc_pattern = re.compile("[\s\n]*[Ii]nclusion [Cc]riteria:?(?: +[Ee]ligibility[ \w]+\: )?(?P<include_crit>[ \n(?:\-|\d)\.\?\"\%\r\w\:\,\(\)]*)[Ee]xclusion [Cc]riteria:?(?P<exclude_crit>[\w\W ]*)")






class CTDoc:
  def __init__(self):
    self.id = None
    self.include_criteria = None
    self.exclude_criteria = None
    self.raw_text = None
    


#----------------------------------------------------------------#
# Document Processing
#----------------------------------------------------------------#






def process_data(
    zip_data, 
    write_file, 
    concat=False, 
    counts=None, 
    max_trials=1e7, 
    start=-1, 
    add_ents=True, 
    mnegs=True, 
    expand=True,
    remove_stops=True,
    id_to_print="", 
    get_only=None
  ):
  """
    zip_data:          str or path ending with .zip containing ct xml files (zipped)
    write_file:        path to write jsonl output
    concat:            bool, whether to concatenate al the grab_only fields into the contents field
    counts:            default None, otherwise dict[str, int], can be passed for updating general statistics about the data
    max_trials:        max number to get, useful for debugging and testing!
    start:             useful if your process gets interrupted and you don't want to start at the begining.
    add_ents:          bool, whether to get entitites with spaCY over the include, exclude criteria (once extracted)
    mnegs:             bool, whether to move negations
    expand:            bool, whether to expand terms in eligibility criteria, makes new alias_crits fields if True
    id_to_print:       str, for debugging, prints doc containing the id given  
    get_only:          list of strings, user can select which fields to grab, otherwise all fields grabbed 

  """
  processed_files = []
  with ZipFile(zip_data, 'r') as zip:
    i = 0
    with open(write_file, "w") as outfile:
      for ct_file in tqdm(zip.namelist()):
        if ct_file.endswith('xml') and (i < max_trials) and (i > start):
          if (get_only is not None) and (Path(ct_file).name[:-4] not in get_only):
            continue 
          results = process_ct_file(zip.open(ct_file, 'r'), id_to_print)

          if counts:
            counts = get_ct_file_counts(zip.open(ct_file, 'r'), counts) 

          if results is not None:
            if concat:
              results = concatenate_data(results)
            
            results = make_content_field(results)

            
            if remove_stops:
              remove_stopwords(results)

            if add_ents:
              results = add_entities(results)

              if mnegs:
                move_neg(results, inc_or_exc="inc")
                move_neg(results, inc_or_exc="exc")

              if expand:
                move_aliases(results, "include")
                move_aliases(results, "exclude")



            json.dump(results, outfile)
            outfile.write("\n")
            processed_files.append(results)

        i += 1
          
  if counts is not None:
    plot_counts(missfld_counts, emptfld_counts)

  print(f"Total nunmber of files processed: {len(processed_files)}")
  return processed_files if counts is None else (processed_files, counts)

  
  if counts is not None:
    plot_counts(missfld_counts, emptfld_counts)

  print(f"Total nunmber of files processed: {len(processed_files)}")
  return processed_files if counts is None else (processed_files, counts)




def process_ct_file( xml_filereader, id_to_print=""):
  """
  xml_filereader:  specific type of object passed from process_data(),
                    used by parsexml library etree.parse() to get tree,
                    allows for easy searching and getting of desired fields 
                    in the document.

  id_to_print:     for debugging a particular CT file, pass the id you wish
                    to have the contents printed for
                    
  desc:            parses xml tree for the set of required fields you see below. 
                    some fields require special treatment. chief among these is
                    'eligibility/criteria/textblock', requiring special 
                    functions to process ths value. 

  returns:         processed dict of required fields as new document
  
  """
  doc_tree = etree.parse(xml_filereader)
  root = doc_tree.getroot()

  docid = root.find('id_info/nct_id').text
  if docid == id_to_print:
    print(etree.tostring(root, pretty_print=True, encoding='unicode'))


  # adding new keys vs subdictionaries?????
  required_fields = {"id":None, 
                    "brief_title":None, 
                    "eligibility/criteria/textblock":{"raw_text":"", "include_criteria":[], 
                                                      "exclude_criteria": []}, 
                    "eligibility/gender":"All", 
                    "eligibility/minimum_age": 0., 
                    "eligibility/maximum_age": 999., 
                    "detailed_description/textblock":None, 
                    "condition":None,
                    "condition/condition_browse":None,
                    "intervention/intervention_type":None,
                    "intervention/intervention_name":None,
                    "intervention_browse/mesh_term":None,
                    "brief_summary/textblock":None,
                  }


  text_fields = ["brief_summary/textblock", 
                "detailed_description/textblock"]

  multiple_entry_fields =  ['condition', 
                            'intervention/intervention_type', 
                            'intervention/intervention_name',  
                            "condition/condition_browse"]

  for field in required_fields.keys():
    try:
      if field in multiple_entry_fields:
        field_val = [result.text for result in root.findall(field)]
        required_fields[field] = field_val
      else:
        if field == 'id':
          field_tag = 'id_info/nct_id'
        else:
          field_tag = field
        field_val = root.find(field_tag).text
        if not empty_pattern.fullmatch(field_val):
          if field == 'eligibility/criteria/textblock':
            inc_elig, exc_elig = process_eligibility_naive(field_val)

            if root.find('id_info/nct_id').text == id_to_print:
              print("\n\nINC CRIT")
              for ic in inc_elig:
                print(ic)
              print("\nEXC CRIT")
              for ec in exc_elig:
                print(ec)

            required_fields['eligibility/criteria/textblock']["include_criteria"] = inc_elig
            required_fields['eligibility/criteria/textblock']["exclude_criteria"] = exc_elig
            required_fields['eligibility/criteria/textblock']['raw_text'] = field_val
          elif field in text_fields:
            required_fields[field] = clean_sentences([field_val])
          elif field.endswith('age'):
            processed_age = process_age_field(field_val)
            if processed_age is not None:
              required_fields[field] = processed_age
          else:
            required_fields[field] = clean_sentences([field_val])[0]


    except:
      # at least one empty or missing field
      pass

  
  return required_fields



def make_content_field( doc):
  """
  doc:     dict containing clinical trial information
  desc:    transfers the content of the summary field to the new `contents`
            field, if summary exists, otherwise contents field becomes emtpy string
  """
  summary = doc['brief_summary/textblock']
  doc['contents'] = summary[0] if summary is not None else ""
  del doc['brief_summary/textblock']
  return doc




def convert_docs_for_basic_filtering( docs, writefile):
  """
  docs:       dicts containing clinical trial data
  writefile:  location to dump the filtered version to
  desc:       creates smaller, filtered versions of the documents for
              some other types of uses

  """
  new_docs = []
  with open(writefile, 'w') as wf:
    for doc in docs:
      new_doc = {}
      new_doc['nct_id'] = doc['id']
      new_doc['min_age'] = doc['eligibility/minimum_age']
      new_doc['max_age'] = doc['eligibility/maximum_age']
      new_doc['gender'] = doc['eligibility/gender']
      new_doc['include_cuis'] = ' '.join([ent['cui']['val'] for ent in doc['inc_ents']])
      new_doc['exclude_cuis'] = ' '.join([ent['cui']['val'] for ent in doc['exc_ents']])
      new_docs.append(new_doc)
      json.dump(new_doc, wf)
      wf.write('\n')
  return new_docs
  




def convert_age_to_year( age, units):
  """
  age:  string result for the age extracted
  unit: string being either years or months (or some variation of those 2)
  desc: converts string to float, months to years if unit is month
  """
  if age is not None:
    age = float(age)
    if units is not None:
      if 'm' in units.lower():
        age /= 12.
      elif 'w' in units.lower():
        age /= 52. 
  return age 




def process_age_field( field_val):
  """
  desc: helper to call concvert_age_to_year.
        extracts unit and value from passed string taken from age field of doc 
  """
  age_match = age_pattern.match(field_val)
  if age_match is not None:
    age = float(age_match.group('age'))
    units = age_match.group('units')
    return convert_age_to_year(age, units)
  else:
    return None








#----------------------------------------------------------------------------
# Functions for extracting eligibility minimum age/max age when in textblock
#----------------------------------------------------------------------------


def resolve_elig_age_with_text( criteria, processed_fields):
  """
  not including




  """
  min_pattern = re.compile("[Mm]inimum [Aa]ge (?:is )(?P<age>\d+) (?P<units>([Yy]e?a?rs|rs?)|([Mm]o?nths?)).*")
  max_pattern = re.compile("[Mm]aximum [Aa]ge (?:is )(?P<age>\d+) (?P<units>([Yy]e?a?rs|rs?)|([Mm]o?nths?)).*")
  between_pattern = re.compile(".*[Bb]etween (?:the ages of )?(?P<min_age>\d+) (?:(?P<unit1>[Yy]ears|[Mm]o?nths?) )?and (?P<max_age>\d+)(?: (?P<unit2>[Yy]ears|[Mm]o?nths?))?")

  for elig_crit in criteria:

    between_match = between_pattern.match(elig_crit)
    if between_match is not None:
      min_age = between_match.group('min_age')
      min_age_unit = between_match.group('unit1')
      min_age = convert_age_to_year(min_age, min_age_unit)
      if min_age is not None:
        processed_fields['eligibility/minimum'] = min_age


      max_age = between_match.group('max_age')
      max_age_unit = between_match.group('unit2')
      max_age = convert_age_to_year(max_age, max_age_unit)
      if max_age is not None:
        processed_fields['eligibility/minimum'] = max_age
      
      break


    if processed_fields['eligibility/minimum'] == 0:
      match = re.match(min_pattern, elig_crit)
      age =  match.group('age') 
      unit = match.group('unit')
      min_age = convert_age_to_year(age, unit)
      if min_age is not None:
        processed_fields['eligibility/minimum'] = min_age
    
    if processed_fields['eligibility/maximum'] == 999:
      match = re.match(max_pattern, elig_crit)
      age =  match.group('age') 
      unit = match.group('unit')
      max_age = convert_age_to_year(age, unit)
      if max_age is not None:
        processed_fields['eligibility/maximum'] = max_age
    









#----------------------------------------------------------------#
# Getting CUI's
#----------------------------------------------------------------#

  

def get_ents( sent_list, top_N):
  """
  sent_list: list of sentence strings
  top_N:     int directing how many aliaseed terms to get
  desc:      uses spaCy pipeline to get entities and link them to terms,
              adds this information, entities lists of the sentences, 
              as a newa_field to the doc
  """
  new_ent_sents = []
  for sent in sent_list:
    nlp_sent = nlp(sent)
    new_ents = []
    for ent in nlp_sent.ents:
      for umls_ent in ent._.kb_ents:
        new_ent = {}
        new_ent['raw_text'] = ent.text
        new_ent['label'] = ent.label_
        new_ent['start'] = ent.start_char
        new_ent['end'] = ent.end_char
        new_ent['cui'] = {'val':umls_ent[0], 'score':umls_ent[1]}
        aliases = linker.kb.cui_to_entity[umls_ent[0]]._asdict()['aliases']
        new_ent['alias_expansion'] = aliases[:min(len(aliases), top_N)]
        new_ent["negation"] = ent._.negex
        #new_ent['covered_text'] = linker.kb.cui_to_entity[umls_ent[0]]
        new_ents.append(new_ent)
        break   # only get first one
    new_ent_sents.append(new_ents)
    
      
  return new_ent_sents



def add_entities( doc, top_N=2):
  """
  desc:    helper function to add the entities got from get_entities() to the doc
  """
  doc["inc_ents"] = get_ents(doc["eligibility/criteria/textblock"]["include_criteria"], top_N)
  doc["exc_ents"] = get_ents(doc["eligibility/criteria/textblock"]["exclude_criteria"], top_N)
  return doc



def remove_stopwords( doc, stopwords=STOP_WORDS):
  """
  desc:    helper function to add versions of the criteria without stopwords to the doc
  """
  doc['inc_no_stop'] = [filter_stops(sent) for sent in doc["eligibility/criteria/textblock"]["include_criteria"]]
  doc['exc_no_stop'] = [filter_stops(sent) for sent in doc["eligibility/criteria/textblock"]["exclude_criteria"]]



def filter_stops(sent, stopwords=STOP_WORDS):
  return ' '.join([word for word in sent.split() if word.lower() not in STOP_WORDS])



def alias_map( field_type):
  """
  field_type: str for directing key names, depending on include, exclude, or topics
  desc:       returns the appropriate field names for the alias creating process

  """
  if field_type == "include":
    crit_field = "include_criteria"
    ent_field = "inc_ents"
    alias_field = "inc_alias_crits"
  elif field_type == "exclude":
    crit_field = "exclude_criteria"
    ent_field = "exc_ents"
    alias_field = "exc_alias_crits"
  else:
    crit_field = "sents"
    ent_field = "ents"
    alias_field = "aliased_sents"
  return crit_field, ent_field, alias_field

  

def make_alias_doc_field( doc, field_type, alias_field):
  """
  desc: modifies doc to contain the apprpriate fields to
        hold the aliased criteria statements. Topic docs don't have inclusion, ec=xclusion,
        so the aliased sentences go into a new field, 'aliased_sents'
  """
  if field_type == "topic":
    doc['aliased_sents'] = []
  elif "alias_crits" not in doc:
    doc["alias_crits"] = {"inc_alias_crits":[], "exc_alias_crits":[]}
  else:
    doc["alias_crits"][alias_field] = []
  return doc




def move_aliases( doc, field_type, dont_alias=DONT_ALIAS):
  """
  doc:            a dict containing clinical trial data
  field_type:     exclude, include, topic are the 3 expected values for this 
  dont_alias:     globally defined set of terms to not be aliased for notocable
                  domain errors, e.g. the term ER, included in many documents,
                  to endoplasmic reticulum

  desc:           uses entities and location from the spaCy pipeline process,
                  with linked UMLS (CUI) terms, and there associated raw text forms
                  it iserts these values into the sentence in the position where
                  the entitiy is located, preserving the sentence except for the new
                  semantic redundancy. syntactic redundancy is not avoided, as many terms 
                  share value with other aliases, or the portion of the sentence being considered 
                  (entities can sopan multiple tokens), except in the case where these 
                  (possibly near) identical values are adjacent, in which case the last term of the 
                  inserted material will be droped before inserting. 
                  This creates longer expanded sentences;





                  example:

                  pre-expansion:



                  post-expansion:

  """

  crit_field, ent_field, alias_field = alias_map(field_type)
  doc = make_alias_doc_field(doc, field_type, alias_field)
  crits = doc['eligibility/criteria/textblock'][crit_field] if (field_type != "topic") else doc[crit_field]
  ents = doc[ent_field]
  for i, (ent_sent, crit) in enumerate(zip(ents, crits)):
    new_crit = crit
    added = 0
    for ent in ent_sent:
      if (ent['raw_text'].lower()) in dont_alias or  (ent['raw_text'].lower()[:-1] in dont_alias):
        continue

      new_aliases = []
      for alias in ent['alias_expansion']:
        alias = alias.lower().strip(',.?')
        if alias != ent['raw_text'].lower():
          new_aliases.append(alias)
      
      begin = new_crit[:ent['start'] + added]
      if not begin.endswith(' ') and (len(begin) > 0):
        begin += ' '
      end = new_crit[ent['start'] + added:]

      if len(new_aliases) == 0:
        continue

      last_alias_word = new_aliases[-1].split()[-1]
      if len(end) > 0:
        if (end.split()[0] == last_alias_word) or (end.split()[0][:-1] == last_alias_word):
          last_alias = new_aliases[-1].split()
          if len(last_alias) > 1:
            new_aliases[-1] = ' '.join(last_alias[:-1])
          
          else:
            new_aliases = new_aliases[:-1]

      add_part = ' '.join(new_aliases)   
      if len(add_part) > 0:
        add_part = add_part + ' '     

      add_len = len(add_part)
      new_crit = begin + add_part + end
      
      added += add_len    

    if field_type == "topic":
      doc[alias_field].append(new_crit.strip())
    else:
      doc["alias_crits"][alias_field].append(new_crit)


#----------------------------------------------------------------#
# Eligibility Processing
#----------------------------------------------------------------#

def process_eligibility(elig_text):
  """
  desc:  this version is not the default for the package and may work in some 
          cases where the naive version fails. it relies heavily on sometimes 
          complicated regex patterns

  """
  # TODO: write tests 

  if elig_text is not None:
    if re.search('[Ii]nclusion [Cc]riteria:[^\w]+\n', elig_text):
      if re.search('[Ee]xclusion Criteria:[^\w]+\n', elig_text):
        inc_raw_text, exc_raw_text = both_inc_and_exc_pattern.match(elig_text).groups()
        include_criteria = clean_sentences(re.split('\-  ', inc_raw_text))
        exclude_criteria = clean_sentences(re.split('\-  ', exc_raw_text))

        if len(exclude_criteria) == 0:
          num_sep_include_criteria = clean_sentences(re.split('[1-9]\.  ', inc_raw_text))
          num_sep_exclude_criteria = clean_sentences(re.split('[1-9]\.  ', exc_raw_text))
          if len(num_sep_exclude_criteria) > 0:
            include_criteria = num_sep_include_criteria
            exclude_criteria = num_sep_exclude_criteria

        include_criteria = check_sentences(include_criteria)
        exclude_criteria = check_sentences(exclude_criteria)
        return (include_criteria, exclude_criteria)

      else:
        include_criteria = inc_only_pattern.match(elig_text).groups(0)[0]
        include_criteria = re.split('\-  +', include_criteria)
        include_criteria = check_sentences(clean_sentences(include_criteria))
        return (include_criteria, [])

    elif re.search('[Ee]xclusion [Cc]riteria:[^\w]+\n', elig_text):
      exclude_criteria = exc_only_pattern.match(elig_text).groups(0)
      exclude_criteria = check_sentences(clean_sentences(re.split('\-  +', exclude_criteria)))
      return ([], exclude_criteria)
    
    else:
      return naive_split_inc_exc(get_sentences(clean_sentences([elig_text])[0]))



def process_eligibility_naive(elig_text):
    """
    elig_text:    a block of raw text like -

      Inclusion Criteria:

        -  Men and women over the age of 18

        -  Skin lesion suspected to either BCC or SCC etc

        -  Patient was referred for biopsy diagnostic/therapeutic before hand, and regardless of
            confocal microscope examination, according to the clinical consideration of physician

      Exclusion Criteria:

        -  Pregnant women

        -  Children

    This script uses regex patterns to split this into inclusion and exclusion, 
    gets the sentences, cleans them, checks them for whether they contain 
    any information once extracted and cleaned, in which case they will be removed.



    """
    inc_crit, exc_crit = [], []
    for h, chunk in enumerate(re.split(r'(?:[Ee]xclu(?:de|sion))|(?:[Ii]neligibility) [Cc]riteria:?', elig_text)):
      for s in re.split(r'\n *(?:\d+\.)|(?:\-) ', chunk):
        if h == 0:
          inc_crit.append(s)
        else:
          exc_crit.append(s)
    
    clean_inc = clean_sentences(inc_crit)
    clean_exc = clean_sentences(exc_crit)
    return check_sentences(clean_inc), check_sentences(clean_exc)



def naive_split_inc_exc(sent_list):
  """
  sent_list:   list of sentence strings
  desc:        given a list of sentences, determines where to partition 
                into inclusion, exclusion criteria solely by the presence of 
                the string that the exclusion matern matches against.
                Note: some docs contain only inclusion criteria, others only 
                exclusion criteria, most both. This version is used in the more 
                complicated splitting procedure above, and is not default.
  """
  i = 0
  while (i < len(sent_list)) and (re.match("[Ee]xclu(?:sion|ded)", sent_list[i]) is None):
    i += 1
  return sent_list[:max(1,i)], sent_list[min(len(sent_list), i):]






#----------------------------------------------------------------#
# Utils
#----------------------------------------------------------------#

def get_sentences( textblock):
  """
  uses spaCy en_core_sci_md package to do sentence segmentation 
  """
  return [s.text for s in nlp(textblock).sents]


def clean_sentences( sent_list):
  """
  sent_list:   list of sentence strings
  desc:        removes a bunch of large spaces and newline characters from the text 
  """
  return [re.sub(r"  +", " ", re.sub(r"[\n\r]", "", s)).strip() for s in sent_list]
  

def check_sentences( sents, words_to_remove=REMOVE_WORDS):
  """
  sents:   a list of strings (not tokenized) representing sentences
  desc:    removes sentences that don't contain any actual criteria
  returns: a list of sentences without filler information (not necessarily one criteria per sent however)
  """
  #include_pattern = re.compile(".*(?:(?:(?:[Ee]|[Ii])(?:(?:x|n)(?:clu(?:(?:de)|(?:sion))))|(?:(?:ne)?ligibility))(?: criteria)? (.*)")
  new_sents = []
  for sent in sents:
    crit = ' '.join([w for w in sent.split() if (w.lower().strip('\:') not in words_to_remove)])
    if len(crit) > 2:
      new_sents.append(crit)
  return new_sents

# only works for lists, dicts, and strings

def data_to_str( data, contents_ignore_fields, grab_only_fields):
  c = ""
  if type(data) == list:
    for d in data:
      c += " " + data_to_str(d, contents_ignore_fields, grab_only_fields)
  elif type(data) == dict:
    for f, v in data.items():
      if len(grab_only_fields) != 0:
        if f in grab_only_fields:
          c += " " + data_to_str(v, contents_ignore_fields, grab_only_fields)
      elif f not in contents_ignore_fields:
        c += " " + data_to_str(v, contents_ignore_fields, grab_only_fields)
  elif (type(data) == float) or (type(data) == int):
    c += " " + str(data) 
  elif type(data) == str:
    c += " " + data
  return c



def concatenate_data( results, ignore_fields = [], grab_only_fields = [], lower=False):
    """
    results:  dictionary of string key, string, list, or dict values
    returns:  2 item dictionary of 'id', 'content' fields, where id is preserved
              but all other values get concatenated into a single string value 
              for 'contents' 
    """
    new_results = {'id':None, 'contents':None}
    contents = ""
    for field, value in results.items():
      if field == 'id':
        new_results['id'] = value
      else:
        if len(grab_only_fields) != 0:
          if field in grab_only_fields:
            contents += data_to_str(value, ignore_fields, grab_only_fields)
        
        elif (field not in ignore_fields):
          contents += data_to_str(value, ignore_fields, grab_only_fields)
        
    contents = contents.strip() if not lower else contents.strip().lower()
    new_results['contents'] = re.sub('    ', ' ', contents)
    return new_results



def get_word_counts( docs):
  """
  docs:       list of dicts contasining clinical trial data
  """
  all_words = []
  for doc in docs:
    for inc_sent in doc["eligibility/criteria/textblock"]['include_criteria']:
      all_words += [w.lower() for w in inc_sent.split()]
    
    for exc_sent in doc["eligibility/criteria/textblock"]['exclude_criteria']:
      all_words += [w.lower() for w in exc_sent.split()]

  return Counter(all_words)


def filter_out_stop( word_counts):
  """
  word_counts:     dict[str, int]
  desc:            removes a globally defined fixed set of words from spaCy library from word count dicts
  """
  for word in STOP_WORDS:
    if word in word_counts:
      del word_counts[word]
  return word_counts



def save_docs_jsonl( docs, writefile):
  """
  desc:    iteratively writes contents of docs as jsonl to writefile 
  """
  with open(writefile, "w") as outfile:
    for doc in docs:
      json.dump(doc, outfile)
      outfile.write("\n")






def get_processed_docs(proc_loc):
  """
  proc_loc:    str or path to location of docs in jsonl form
  """
  with open(proc_loc, 'r') as json_file:
    json_list = list(json_file)

  return [json.loads(json_str) for json_str in json_list]




def remove_top_words( text, remove_words):
  """
  text:         str of text to be filtered
  remove_words: set of most common strings 
  """
  new_contents = [word for word in text.split() if word not in remove_words]
  return ' '.join(new_contents)



def concat_all( docs, writefile, ignore_fields=[], grab_only_fields=[], lower=False, top_words=None):
  """
  docs:               list of dictionary objects containing CT data processed from zipfile
  writefile:          str or path object where dictionaries are written as jsonl (one dict per line)
  ignore_fields:      list of strings representing keys to be ignoredwhen writing
  grab_only_fields:   opposite of ignore_fields, keys to get
  lower:              boolean saying where to call lower() on all string data 
  top_words:          set of words to be removed
  desc:               iterates through doc list, calls conatenate_data() on each doc,
                      creating a new 'contents' field from the concatenated fields
                      with passed args, writes contents selected to writefile, 
                      one dict per line (jsonl)


  """
  new_docs = copy.copy(docs)
  with open(writefile, 'w') as wf:
    for doc in new_docs:
      concat_doc = concatenate_data(doc, ignore_fields=ignore_fields, grab_only_fields=grab_only_fields, lower=lower)
      doc['contents'] = concat_doc['contents']
      if top_words is not None:
        doc['contents'] = remove_top_words(doc['contents'], top_words)
      json.dump(doc, wf)
      wf.write('\n')
  return new_docs









if __name__ == "__main__":
    print("hello")