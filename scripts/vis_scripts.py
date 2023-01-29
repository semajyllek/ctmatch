
from typing import Dict, List, NamedTuple
from ctproc import CTDocument, EligCrit
from matplotlib import pyplot as plt
from collections import defaultdict
from zipfile import ZipFile
from lxml import etree
import pandas
import tqdm
import re

from utils import *

class FieldCounter(NamedTuple):
  missfld_counts: Dict[str, int] = defaultdict(int)
  emptfld_counts: Dict[str, int] = defaultdict(int)
  elig_form_counts: Dict[str, int] = defaultdict(int)
  unit_counts: Dict[str, int] = defaultdict(int)



#----------------------------------------------------------------#
# EDA Utility Functions
#----------------------------------------------------------------#

# viewing
def print_elig_result(doc, dont_print=[]):
  for k, v in doc.elig_crit.__dict__.items():
    if k in dont_print:
      continue
    if type(v) == list:
      print('\n' + k)
      for v_i in v:
        print(v_i)
    else:
      print(f"{k}: {v}")


def display_elig(docs: List[CTDocument]) -> None:
  age_in_elig_text_dist = count_elig_crit_age_in_text(docs)
  total = sum(age_in_elig_text_dist.values())
  print(f"{total} out of {len(docs)} documents had age in eligibility text: {total / len(docs)}%")

  age_in_elig_counts_df = pandas.DataFrame(age_in_elig_text_dist, index=[0])
  age_in_elig_counts_df.plot(kind="bar", xticks=[], xlabel="include_or_exclude", ylabel="count", title="Age in Eligibility Criteria Text Distribution")
  print(age_in_elig_counts_df)
  inc_ratio = age_in_elig_text_dist['inc_ct'] / total
  exc_ratio = age_in_elig_text_dist['exc_ct'] / total
  print(f"{age_in_elig_text_dist['inc_ct']} instances in inclusion statements ({inc_ratio}%), {age_in_elig_text_dist['exc_ct']} instances in exclusion statements ({exc_ratio}%)")





def print_ent_sent(ent_sent):
  for e in ent_sent:
    e_small = {}
    e_small['raw_text'] = e['raw_text']
    e_small['start'] = e['start']
    e_small['end'] = e['end']
    e_small['negation'] = e['negation']
    print(e_small.items())






#--------------------------------------------------------------------------------------#
# methods for getting counts
#--------------------------------------------------------------------------------------#

def process_counts(zip_data: str) -> FieldCounter:
  """
  desc:       main method for processing a zipped file of clinical trial XML documents from clinicaltrials.gov
              parameterized by CTConfig the self ClinProc object was initialized with
  returns:    yields processed CTDocuments one at a time
  """

  counts = FieldCounter()
  with ZipFile(zip_data, 'r') as zip_reader:
    for ct_file in tqdm(zip_reader.namelist()):
      if not ct_file.endswith('xml'):
        continue

      counts = get_ct_file_counts(zip_reader.open(ct_file), counts)
  return counts




def get_ct_file_counts(xml_filereader, counts: FieldCounter) -> FieldCounter:
  doc_tree = etree.parse(xml_filereader)
  root = doc_tree.getroot()

  # adding new keys vs subdictionaries?????
  required_fields = {
    "id":None, 
    "brief_title":None, 
    "eligibility/criteria/textblock":None, 
    "eligibility/gender":"Default Value", 
    "eligibility/minimum_age":{"male":0, "female":0}, 
    "eligibility/maximum_age":{"male":999., "female":999.}, 
    "detailed_description/textblock":None, 
    "condition":None,
    "condition/condition_browse":None,
    "intervention/intervention_type":None,
    "intervention/intervention_name":None,
    "intervention_browse/mesh_term":None,
    "brief_summary/textblock":None,
  }
        
  for field in required_fields.keys():
    field_tag = 'id_info/nct_id'  if field == 'id' else field
    try:
      field_val = root.find(field_tag).text
      if not EMPTY_PATTERN.fullmatch(field_val):
        if field == 'eligibility/criteria/textblock':
          counts.elig_form_counts = get_elig_counts(field_val, counts.elig_form_counts)
        elif "age" in field:
          age_match = AGE_PATTERN.match(field_val)
          if age_match is not None:
            unit = age_match.group('units')
            if unit is not None:
              counts.unit_counts[unit] += 1



    except:
      if root.find(field_tag) is None:
        counts.missfld_counts[field]  += 1
      elif EMPTY_PATTERN.fullmatch(root.find(field_tag).text):
        counts.emptfld_counts[field]  += 1
    
  return counts












def get_elig_counts(elig_text: str, elig_form_counts: Dict[str, int]) -> Dict[str, int]:
  assert elig_text is not None, "Eligibility text is empty"
  if re.search('[Ii]nclusion [Cc]riteria:[^\w]+\n', elig_text):
    if re.search('[Ee]xclusion Criteria:[^\w]+\n', elig_text):
      elig_form_counts["inc_and_exc"] += 1
      return elig_form_counts
    else:
      elig_form_counts["inc_only"] += 1
      return elig_form_counts

  elif re.search('[Ee]xclusion [Cc]riteria:[^\w]+\n', elig_text):
    elig_form_counts["exc_only"] += 1
    return elig_form_counts
  
  else:
    elig_form_counts["textblock"] += 1
    return  elig_form_counts




def get_counts(docs: List[CTDocument]):
  gender_dist = defaultdict(int)
  min_age_dist = defaultdict(int)
  max_age_dist = defaultdict(int)
  for doc in docs:
    gender_dist[doc.elig_gender] += 1
    min_age_dist[doc.elig_crit.elig_min_age] += 1
    max_age_dist[doc.elig_max_age] += 1
  return gender_dist, min_age_dist, max_age_dist
  


def get_relled(topic_id, rel_dict):
  twos, ones, zeros = set(), set(), set()
  for doc_id, rel in rel_dict[topic_id].items():
    if rel == 1:
      ones.add(doc_id)
    elif rel == 2:
      twos.add(doc_id)
    else:
      zeros.add(doc_id)
  return {"twos": twos, "ones": ones, "zeros": zeros}
  
def scan_for_age(
  elig_crit: EligCrit, 
  inc_or_ex: str = 'include'
) -> bool:
  crit_to_scan = elig_crit.include_criteria if inc_or_ex == 'include' else elig_crit.exclude_criteria
  for crit in crit_to_scan:
    if re.match(r' ages? ', crit.lower()) is not None:
      return True
  return False


def count_elig_crit_age_in_text(docs, skip_predefined:bool = True):
  age_in_elig_text_dist = defaultdict(int)
  skipped = 0
  for doc in docs:
    if skip_predefined:
      if (doc.elig_min_age != 0) or (doc.elig_max_age != 999):  # author(s) have specified SOME criteria, assumes judgment prefers this field to free trex in criteria textblock
        skipped += 1
        continue

    age_in_elig_text_dist['include'] += scan_for_age(doc.elig_crit, 'include')
    age_in_elig_text_dist['exclude'] += scan_for_age(doc.elig_crit, 'exclude') 
    
  print(f"Total skipped: {skipped}")
  return age_in_elig_text_dist




def get_missing_criteria(docs: List[CTDocument]):
  missing_inc_ids, missing_exc_ids = {}, {}
  for d in docs:

    if len(d.elig_crit.include_criteria) == 0:
      missing_inc_ids.add(d.nct_id)
  
    if len(d.elig_crit.exclude_criteria) == 0:
      missing_exc_ids.add(d.nct_id)
      
  return missing_inc_ids, missing_exc_ids


# for evaluating effect of filtering
def get_doc_percent_elig(filtered_docs_by_topic: Dict[str, set]):
  percents_elig = []
  for topic_id, doc_list in filtered_docs_by_topic.items():
    per = len(doc_list) / 3262.0
    percents_elig.append(per)
    print(topic_id, len(doc_list), per)
  mean_elig = sum(percents_elig) / len(percents_elig)
  print(f"Mean elgibile number of docs: {mean_elig}")





# plotting

def plot_counts(missfld_counts, emptfld_counts):
  miss_df = pandas.DataFrame(missfld_counts, index=[0])
  miss_df.plot(kind='bar', xticks=[], title="Missing Fields", ylabel="count", xlabel="field")
  plt.legend(loc=(1.04, 0))

  empt_df = pandas.DataFrame(emptfld_counts, index=[0])
  empt_df.plot(kind='bar', xticks=[], title="Empty Fields", ylabel="count", xlabel="field")
  plt.legend(loc=(1.04, 0))




#----------------------------------------------------------------#
# EDA Test Data Utility Functions
#----------------------------------------------------------------#


def get_test_rels(test_rels):
    rel_dict = defaultdict(lambda:defaultdict(int))
    rel_type_dict = defaultdict(int)
    for line in open(test_rels, 'r').readlines():
        topic_id, zero, doc_id, rel = re.split(r'\s+', line.strip())
        rel_dict[topic_id][doc_id] = int(rel)
        rel_type_dict[rel] += 1
    return rel_dict, rel_type_dict

def analyze_test_rels(test_rels_path):
    rel_dict, rel_type_dict = get_test_rels(test_rels_path)

    print("Rel Type Results:")
    for t, n in rel_type_dict.items():
      print(t + ': ' + str(n))

    lengths = dict()
    all_qrelled_docs = set()
    for tid in rel_dict.keys():
        lengths[tid] = len(rel_dict[tid])
        for d in rel_dict[tid].keys():
            all_qrelled_docs.add(d)
    for topic, num_relled in lengths.items():
        print(topic, num_relled)
    print(f"Total relled: {len(all_qrelled_docs)}")
    return rel_type_dict, rel_dict, all_qrelled_docs






if __name__ == '__main__':
	qrels_path = '/Users/jameskelly/Documents/cp/ctmatch/data/qrels-clinical_trials.txt'
	rel_type_dict, rel_dict, all_qrelled_docs = analyze_test_rels(qrels_path)
  #docs_path = '/Users/jameskelly/Documents/cp/ctproc/clinicaltrials.gov-16_dec_2015_17.zip'
  #counts = process_counts(docs_path)