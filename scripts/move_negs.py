

def check_if_bad_end(sent, end_counts=None):
  bad_ends = ["be", "have", "to", "with", "for", "of", "no", "not", "other", "been"]
  last_word = sent.split()[-1]
  if end_counts is not None:
    end_counts[last_word] += 1
  for be in bad_ends:
    if last_word == be:
      return True
  return False



def get_good_end(sent):
  good_ends = ["severe", "significant", "current", "prior", "previous"]
  last_word = sent.strip().split()[-1]
  return last_word
  #for ge in good_ends:
  #  if last_word == ge:
  #    return last_word
  #return None




def move_neg(doc, inc_or_exc="inc", end_counts=None):
  new_crits = []
  new_ent_sents = []
  changes = 0

  if "moved_negs" not in doc:
    doc["moved_negs"] = {"include_criteria":doc['eligibility/criteria/textblock']['include_criteria'],
                       "exclude_criteria":doc['eligibility/criteria/textblock']["exclude_criteria"], 
                       "inc_ents":doc['inc_ents'], "exc_ents":doc["exc_ents"]}

  # allows moving negation from inc to exc or vice versa
  if inc_or_exc == 'inc':
    crit_field = "include_criteria"
    ent_field = "inc_ents"
    other_crit_field = "exclude_criteria"
    other_ent_field = "exc_ents"
  else:
    crit_field = "exclude_criteria"
    ent_field = "exc_ents"
    other_crit_field = "include_criteria"
    other_ent_field = "inc_ents"


  # this loops got verbose because I had to keep track of both the positions in the criteria string and
  # the entities list in order to build the representations desired
  for crit, ent_sent in zip(doc['eligibility/criteria/textblock'][crit_field], doc[ent_field]):
    i = 0
    neg_found = False
    split_crit = crit.split()
    while i < len(ent_sent):
      ent = ent_sent[i]
      ent_start = i
      if ent['negation']:
        neg_found = True
        text_start = ent['start']                      # find the place in the criteria string where the negated entitiy raw string starts
        
        while (i < len(ent_sent) - 1) and ent_sent[i+1]['negation']: # get all of the text and relative ents associated with that negated span
          i += 1

        text_end = ent_sent[i]['end']
        ent_end = i + 1

        # stuff for the beginning and end of remaming string and corresponding ent lists
        end_part = crit[text_end:]
        start_part = crit[:text_start]
        ent_start_part = ent_sent[:ent_start]
        ent_end_part = ent_sent[ent_end:]
        new_crit = ""
        new_ent_sent = []
        other_ent = []



        bad_end = True
        if (len(start_part.strip().split()) > 3) and not check_if_bad_end(start_part, end_counts):
          new_crit += start_part
          new_ent_sent += ent_start_part
          bad_end = False


        other_crit = crit[text_start:text_end]

        if not bad_end:
          good_end = get_good_end(start_part)
        else:
          good_end = None
        if good_end is not None:
          other_crit = good_end + ' ' + other_crit # could be severe, significant, etc. 
        other_ent = ent_sent[ent_start:ent_end]
     
        if len(end_part.strip().split()) > 2:
          new_crit += end_part
          new_ent_sent += ent_end_part

        if len(other_ent) > 1:
          doc['moved_negs'][other_ent_field].append(other_ent)
          doc['moved_negs'][other_crit_field].append(other_crit)
      

      i += 1


    if neg_found:
      changes += 1
      if len(new_crit.split()) > 2:
        new_crits.append(new_crit)
        new_ent_sents.append(new_ent_sent)
    else:
      new_crits.append(crit)
      new_ent_sents.append(ent_sent)

 
  
  doc['moved_negs'][crit_field] = new_crits
  doc['moved_negs'][ent_field] = new_ent_sents
  return doc, changes





def move_negs(docs, inc_or_exc="inc", end_counts=None):
  total_changes = 0
  total_docs_changed = 0
  for doc in docs:       
    _, count = move_neg(doc, inc_or_exc=inc_or_exc, end_counts=end_counts)
    if count > 0:
      total_changes += count
      total_docs_changed += 1
      
  return total_docs_changed, total_changes

  




