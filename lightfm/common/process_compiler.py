import re 
# 'experiments/ex4/preprocess/connect.py'
def block_str_generator(file_name):
    f = open(file_name)
    content = "".join(list(f))
    f.close()
    content = _clean_up(content)
    code_lines = content.split('\n')
    blocks = _generate_code_blocks(code_lines)
    tmp_assigned_blocks = _cat_conn_and_tmp_file(blocks)
    return conn_str_checking(tmp_assigned_blocks)

def conn_str_checking(block_gen):
    for i, block in enumerate(block_gen):
        try:
            if type(block) == str:
                assert '=' in block
            elif type(block) == tuple:
                assert '=' in block[0]
            yield block 
        except:
            raise ValueError(f"Error happend at the {i+1}th connection:", block)

def _cat_conn_and_tmp_file(block_str_gen):
    duo_queue = []
    for block in block_str_gen:
        assert len(duo_queue) <= 2
        if len(duo_queue) == 0:
            duo_queue.append(block)
        else:
            if '=' not in block and block.split('(')[0] == '': # is assignment of tmp file name 
                result_dirs = eval(block)
                if type(result_dirs) == tuple:
                    result_dirs = list(result_dirs)
                yield duo_queue.pop(0), result_dirs
            else:
                duo_queue.append(block)
                yield duo_queue.pop(0)
    for remain_block in duo_queue:
        yield remain_block

def _clean_up(content):
    # remove \ + newline 
    content = content.replace('\\\n', '') 
    # remove multiline comment surrounded by '''''' or """"""
    content = re.sub(
        r"'''((.|\n)*)'''", '', content)
    content = re.sub(
        r'"""((.|\n)*)"""', '', content)
    # remove comment
    content = re.sub(r'#.*','', content)
    return content

def _generate_code_blocks(code_lines):
    # code_blocks = []
    tmp_str = ''
    for code_line in code_lines:
        if _has_last_parenthesis(code_line): 
            tmp_str += code_line 
            yield tmp_str
            tmp_str = ''
        else:
            tmp_str += code_line
            
def _has_last_parenthesis(code_line):
    return bool(re.search('\)( *)', code_line))

def handle_command(code_line):
    if '#' in code_line:
        first_command_mark_index = code_line.index('#')
        return code_line[:first_command_mark_index] 
    else:
        return code_line