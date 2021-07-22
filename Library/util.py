import re
from IPython.core.display import display, HTML

def jupyter_wide_screen():
    display(HTML("<style>.container { width:100% !important; }</style>"))

def strings_contain_words(strings, words):
    
    for word in words:
        lambda_func = lambda x: word.lower() in x.lower()
        strings = list(filter(lambda_func, strings))
        
    return strings

def strings_contain_patterns(strings, patterns, ignore_case=False):
    
    for pattern in patterns:
        if ignore_case:
            re_pattern = re.compile(pattern, flags=re.I)
        else:
            re_pattern = re.compile(pattern)
        lambda_func = lambda string: re_pattern.search(string) is not None
        
        strings = list(filter(lambda_func, strings))
        
    return strings

def add_unique_entry(target_list, source_list):
    for entry in source_list:
        if entry not in target_list:
            target_list.append(entry)