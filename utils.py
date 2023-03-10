
from typing import Any, List
import json
import re




#----------------------------------------------------------------#
# global regex patterns for use throughout the methods
#----------------------------------------------------------------#


EMPTY_PATTERN = re.compile('[\n\s]+')
"""
both_inc_and_exc_pattern = re.compile(r\"\"\"[\s\n]*[Ii]nclusion [Cc]riteria:?               # top line of both
                                      (?:[ ]+[Ee]ligibility[ \w]+\:[ ])?                  # could contain this unneeded bit next
                                      (?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)      # this should get all inclusion criteria as a string
                                      [Ee]xclusion[ ][Cc]riteria:?                        # delineator to exclusion criteria
                                      (?P<exclude_crit>[\w\W ]*)                          # exclusion criteria as string
                                      \"\"\", re.VERBOSE)
"""
INC_ONLY_PATTERN = re.compile('[\s\n]+[Ii]nclusion [Cc]riteria:?([\w\W ]*)')
EXC_ONLY_PATTERN = re.compile('[\n\r ]+[Ee]xclusion [Cc]riteria:?([\w\W ]*)')
AGE_PATTERN = re.compile('(?P<age>\d+) *(?P<units>\w+).*')
YEAR_PATTERN = re.compile('(?P<year>[yY]ears?.*)')
MONTH_PATTERN = re.compile('(?P<month>[mM]o(?:nth)?)')
WEEK_PATTERN = re.compile('(?P<week>[wW]eeks?)')

BOTH_INC_AND_EXC_PATTERN = re.compile("[\s\n]*[Ii]nclusion [Cc]riteria:?(?: +[Ee]ligibility[ \w]+\: )?(?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)[Ee]xclusion [Cc]riteria:?(?P<exclude_crit>[\w\W ]*)")


# -------------------------------------------------------------------------------------- #
# I/O utils
# -------------------------------------------------------------------------------------- #

def save_docs_jsonl(docs: List[Any], writefile: str) -> None:
  """
  desc:    iteratively writes contents of docs as jsonl to writefile 
  """
  with open(writefile, "w") as outfile:
    for doc in docs:
      json.dump(doc, outfile)
      outfile.write("\n")



def get_processed_docs(proc_loc: str):
  """
  proc_loc:    str or path to location of docs in jsonl form
  """
  with open(proc_loc, 'r') as json_file:
    json_list = list(json_file)

  return [json.loads(json_str) for json_str in json_list]

