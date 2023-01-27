

from matplotlib import pyplot as plt
from collections import defaultdict
from typing import Dict
from lxml import etree
import pandas
import re





empty_pattern = re.compile('[\n\s]+')
"""
both_inc_and_exc_pattern = re.compile(r\"\"\"[\s\n]*[Ii]nclusion [Cc]riteria:?               # top line of both
                                      (?:[ ]+[Ee]ligibility[ \w]+\:[ ])?                  # could contain this unneeded bit next
                                      (?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)      # this should get all inclusion criteria as a string
                                      [Ee]xclusion[ ][Cc]riteria:?                        # delineator to exclusion criteria
                                      (?P<exclude_crit>[\w\W ]*)                          # exclusion criteria as string
                                      \"\"\", re.VERBOSE)
"""
inc_only_pattern = re.compile('[\s\n]+[Ii]nclusion [Cc]riteria:?([\w\W ]*)')
exc_only_pattern = re.compile('[\n\r ]+[Ee]xclusion [Cc]riteria:?([\w\W ]*)')
age_pattern = re.compile('(?P<age>\d+) *(?P<units>\w+).*')
year_pattern = re.compile('(?P<year>[yY]ears?.*)')
month_pattern = re.compile('(?P<month>[mM]o(?:nth)?)')
week_pattern = re.compile('(?P<week>[wW]eeks?)')

both_inc_and_exc_pattern = re.compile("[\s\n]*[Ii]nclusion [Cc]riteria:?(?: +[Ee]ligibility[ \w]+\: )?(?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)[Ee]xclusion [Cc]riteria:?(?P<exclude_crit>[\w\W ]*)")

# count dictionaries if needed
missfld_counts = {"id_info/nct_id":0, 
                     "brief_title":0, 
                     "eligibility/criteria/textblock":0, 
                     "eligibility/gender":0, 
                     "eligibility/minimum_age":0, 
                     "eligibility/maximum_age":0, 
                     "detailed_description/textblock":0, 
                     "condition":0,
                     "condition/condition_browse":0,
                     "intervention/intervention_type":0,
                     "intervention/intervention_name":0,
                      "intervention_browse/mesh_term":0,
                     "brief_summary/textblock":0
                     }


emptfld_counts = {"id_info/nct_id":0, 
                     "brief_title":0, 
                     "eligibility/criteria/textblock":0, 
                     "eligibility/gender":0, 
                     "eligibility/minimum_age":0, 
                     "eligibility/maximum_age":0, 
                     "detailed_description/textblock":0, 
                     "condition":0,
                     "condition/condition_browse":0,
                     "intervention/intervention_type":0,
                     "intervention/intervention_name":0,
                     "intervention_browse/mesh_term":0,
                     "brief_summary/textblock":0
                     }

elig_form_counts = {"inc_and_exc":0,
                    "inc_only":0,
                    "exc_only":0,
                    "textblock":0
                    }

unit_counts = defaultdict(int)







#----------------------------------------------------------------#
# EDA Utility Functions
#----------------------------------------------------------------#

# viewing
def print_elig_result(doc, crit_key='eligibility/criteria/textblock', dont_print=[]):
  for k, v in doc[crit_key].items():
    if k in dont_print:
      continue
    if type(v) == list:
      print('\n' + k)
      for v_i in v:
        print(v_i)
    else:
      if k not in dont_print:
        print(f"{k}: {v}")


def display_elig(docs):
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





# counting

def get_elig_counts(elig_text, elig_form_counts=elig_form_counts):
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


def get_ct_file_counts(xml_filereader, counts):
  missfld_counts, emptfld_counts, elig_form_counts, unit_counts = counts
  doc_tree = etree.parse(xml_filereader)
  root = doc_tree.getroot()


  # adding new keys vs subdictionaries?????
  required_fields = {"id":None, 
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
        
  missing_fields = False
  for field in required_fields.keys():
    field_tag = 'id_info/nct_id'  if field == 'id' else field
    try:
      field_val = root.find(field_tag).text
      if not empty_pattern.fullmatch(field_val):
        if field == 'eligibility/criteria/textblock':
          elig_form_counts = get_elig_counts(field_val, elig_form_counts)
        elif "age" in field:
          age_match = age_pattern.match(field_val)
          if age_match is not None:
            unit = age_match.group('units')
            if unit is not None:
              unit_counts[unit] += 1



    except:
      missing_fields = True
      if root.find(field_tag) is None:
        missfld_counts[field]  += 1
      elif empty_pattern.fullmatch(root.find(field_tag).text):
        emptfld_counts[field]  += 1
    
  return (missfld_counts, emptfld_counts, elig_form_counts, unit_counts)




def get_counts(docs):
  gender_dist = defaultdict(int)
  min_age_dist = defaultdict(int)
  max_age_dist = defaultdict(int)
  for doc in docs:
    eligibility_gender = doc['eligibility/gender']
    eligibility_min_age = doc['eligibility/minimum_age']
    eligibility_max_age = doc['eligibility/maximum_age']
    gender_dist[eligibility_gender] += 1

    min_age_dist[eligibility_min_age] += 1
    max_age_dist[eligibility_max_age] += 1
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





def count_elig_crit_age_in_text(docs, skip_predefined:bool = True):
  age_in_elig_text_dist = defaultdict(int)
  skipped = 0
  for doc in docs:
    if skip_predefined:
      if (doc['eligibility/minimum_age'] != 0) or (doc['eligibility/maximum_age'] != 999):  # author(s) have specified SOME criteria, assumes judgment prefers this field to free trex in criteria textblock
        skipped += 1
        continue 
      
   
    combined_crit = doc['eligibility/criteria/textblock']['include_criteria'] + doc['eligibility/criteria/textblock']['exclude_criteria']
    for crit in combined_crit:
      if (" age " in crit.lower()) or (" ages " in crit.lower()):
        for k, d in doc['eligibility/criteria/textblock'].items():
          for cr in d:
            if cr == crit:
              if 'include' in k:
                age_in_elig_text_dist['inc_ct'] += 1
              else:
                age_in_elig_text_dist['exc_ct'] += 1
        break
  print(f"Total skipped: {skipped}")
  return age_in_elig_text_dist



def get_missing_criteria(docs, min_index=0, max_index=5, crit_key='eligibility/criteria/textblock'):
  missing_inc, missing_exc = 0, 0
  for d in docs:
    elig_doc = d[crit_key]
    if len(elig_doc['include_criteria']) == 0:
      missing_inc += 1
      if (missing_inc > 0) and (missing_inc < max_index):
        print('\nMISSING INCLUDE:')
        print(d['id'])
        for k, v in elig_doc.items():
          if type(v) == list:
            print('\n' + k)
            for v_i in v:
              print(v_i) 
          else:
            print(f"{k}: {v}")
        
        
    if len(elig_doc['exclude_criteria']) == 0:
      missing_exc += 1
      if (missing_exc > min_index) and (missing_exc < max_index):
        print('\nMISSING EXCLUDE:')
        print(d['id'])
        for k, v in elig_doc.items():
          if type(v) == list:
            print( '\n' + k)
            for v_i in v:
              print(v_i) 
          else:
            print(f"\n{k}: {v}")
      
  return missing_inc, missing_exc, len(docs)


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
    


