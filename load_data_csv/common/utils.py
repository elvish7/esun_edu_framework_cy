#!/usr/bin/env python
# coding: utf-8
import os, sys
import autopep8
import re

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

class Str2CodeAdaptor:
    '''
    By inherenting this class, we will be able to :  
    
      1. insert public and private variables into a class by assigning variable and experssion 
      2. add variable into the top-level global variable pool. 
      
    '''
    @staticmethod
    def get_global_var(__globals__, var):
        assert type(var) == str
        return __globals__[var]
    @staticmethod
    def get_var_from_obj(obj, var, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str
        if declare == 'public':
            return eval(f'obj.{var}')
        else:
            return eval(f'obj._{var}')
    @staticmethod
    def get_var(env, var):
        '''
        This function allows the call to `get_global_var` and `get_var_from_obj` with the determination of `env`, 
        where env can be `globals()` or a object. 
        '''
        if type(env) == dict:
            return Str2CodeAdaptor.get_global_var(env, var)
        else:
            return Str2CodeAdaptor.get_var_from_obj(env, var)
    @staticmethod 
    def set_global_var(__globals__, var, value):
        assert type(var) == str 
        assert type(__globals__) == dict
        Str2CodeAdaptor.tmp = value
        __globals__.update({var: Str2CodeAdaptor.tmp})
        del Str2CodeAdaptor.tmp 
    @staticmethod
    def add_var_to_obj(obj, var, value, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str
        tmp = value
        if declare == 'public':
            exec(f'obj.{var}=tmp')
        else:
            exec(f'obj._{var}=tmp')
    @staticmethod
    def add_var(env, var, value):
        '''
        This function allows the call to `set_global_var` and `add_var_to_obj` with the determination of `env`, 
        where env can be `globals()` or a object. 
        '''
        if type(env) == dict:
            Str2CodeAdaptor.set_global_var(env, var, value)
        else:
            Str2CodeAdaptor.add_var_to_obj(env, var, value)
            
    def set_var(self, var, value, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str 
        tmp = value
        if declare == 'public':
            exec(f'self.{var}=tmp')
        else:
            exec(f'self._{var}=tmp')
            
    @staticmethod
    def fix_code(code_str):
        fixed_code = autopep8.fix_code(code_str)
        fixed_code = fixed_code.replace('\n','')
        fixed_code = fixed_code.replace(' ','')
        return fixed_code
    
    @staticmethod
    def breakdown_function_call(code_str):
        '''
        This function takes a python code string of function calling and 
        break it down into a list of output variable names, the function name, and the calling message string to the function. 

        - Input: code_str, a string in the formate of `a, b = function(c, d, e, f = f, g = g)`, 
                            where a, b is the output variables, `function` the function,
                            and `(c, d, e, f = f, g = g)` the calling message string. 
        - Output:
            - output_variables: a list of output variable names.
            - function_str: the function name. 
            - func_input_str: the calling message string to the function. 
            
        - Test Cases: 
            X: test_case = '(a, b) = ((function))(c, d, e, f = f1,   g = g1, g =   g2)'
            O: test_case = '(a, b) = function(c, d, e, f = f1, g = g1, g = g2)'
            O: test_case = 'a, b = function(c, d, e, f = f1  , g = g1, g = g2)'
            O: test_case = 'a = function(c, d, e, f  = f1 ,   g = g1, g = g2)'
        '''
        # Step 1: Fix Code 
        fixed_code = Str2CodeAdaptor.fix_code(code_str)
        assert fixed_code[-1] == ')'

        # Step 2: get output variables 
        output_variables = fixed_code.split("=")[0].replace('(','').replace(')','').split(',')

        # Step 3: get function 
        ## make sure no () around the function 
        first_eq_mark = fixed_code.index("=")
        assert fixed_code[first_eq_mark+1]!='('
        fixed_code[first_eq_mark+1].split('(')[0]
        function_str = fixed_code[first_eq_mark+1:].split('(')[0]

        # Step 4: get function input string : (x, y, z, w = w, v = v, …) 
        func_input_str = fixed_code[first_eq_mark+1+len(function_str):]
        
        return output_variables, function_str, _extract_args(func_input_str), _extract_kwargs(func_input_str)



def _get_first_pair(input_str):
    re_obj = re.search(VAR_VALUE_PAIR, input_str)
    try:
        return re_obj.group(0)[1:-1], re_obj.start()+1, re_obj.end()-1 
    except:
        return None, 0, 0

def _GET_STRUCT_REGEX(PAR_L, PAR_R, TOKEN):
    _STRUCT = f'{PAR_L}{TOKEN}((,{TOKEN})*){PAR_R}' # [xx, xxx, xx]
    _EMPTY_STRUCT = f'{PAR_L}{PAR_R}'
    return f'({_STRUCT}|{_EMPTY_STRUCT})'

def _GET_VAR_VALUE_PAIR_REGEX():
    
    NUMBER = '[0-9.]+'
    
    STR_BOUND1 = '\"'
    STRING1 = f'{STR_BOUND1}[^{STR_BOUND1}]+{STR_BOUND1}'
    STR_BOUND2 = "\'"
    STRING2 = f"{STR_BOUND2}[^{STR_BOUND2}]+{STR_BOUND2}"

    TOKEN = f'({VARNAME}|{NUMBER}|{STRING1}|{STRING2})'

    ARRAY = _GET_STRUCT_REGEX('\[', '\]', TOKEN)
    SET = _GET_STRUCT_REGEX('\{', '\}', TOKEN)
    TUPLE = _GET_STRUCT_REGEX('\(', '\)', TOKEN)
    DICT = _GET_STRUCT_REGEX('\{', '\}', f'{TOKEN}:{TOKEN}')

    ASSIGNMENT = f'({VARNAME}|{NUMBER}|{STRING1}|{STRING2}|{ARRAY}|{SET}|{TUPLE}|{DICT})'

    VAR_VALUE_PAIR = f'(,|\(){VARNAME}={ASSIGNMENT}(,|\))'
    
    return VAR_VALUE_PAIR

VARNAME = '([a-zA-Z_0-9]+)'
VAR_VALUE_PAIR = _GET_VAR_VALUE_PAIR_REGEX()

def _extract_args(input_str):
    return _get_all(input_str, _get_first_arg)


def _extract_kwargs(input_str):
    input_pair = []
    for assignment in _get_all(input_str, _get_first_pair):
        eq_index = assignment.index("=")
        var_name = assignment[:eq_index]
        value = assignment[eq_index+1:]
        input_pair.append((var_name, value))
    input_dict = dict(input_pair)
    return input_dict

def _get_all(input_str, search_func):
    next_str = input_str
    ans_list = []
    while True:
        ans, s, e = search_func(next_str)
        if not ans:
            break
        ans_list.append(ans)
        next_str = next_str[e:]
    return ans_list



def _get_first_arg(input_str):
    ARG = f'(,|\(){VARNAME}(,|\))'
    re_obj = re.search(ARG, input_str)
    try:
        return re_obj.group(0)[1:-1], re_obj.start()+1, re_obj.end()-1 
    except:
        return None, 0, 0


'''
Test:
for test_example in [
    '(a=2,b=3,c=44,d=e4)',
    '(a=2,b=3,c=[],d=12,f=3.3,h=12.,g=.12)',
    '(a="2",b="3",c="[]",d="12",f="3.3",h="12.",g=".12")',
    "(a='2',b='3',c='[]',d='12',f='3.3',h='12.',g='.12')",
    "(a='2',b='3',c=[],d=['12',a,ab,a5,1,1.1,3.53,.124])",
    "(a={},b={1,2,3},e={f:'f',g:'g'},c=[],d=['12',a,ab,a5,1,1.1,3.53,.124])",
    "(a=(),b=(1,2,3),e=(1),e=(a,b,cc,d3))",
    "(a={},b={'a':a,'v':31,12:0},e=(1),e=(a,b,cc,d3)),f=1)",
]:
    print(test_example)
    input_str = test_example # remove the outer () in the beginning 
    ans = _breakdown_function_input_str(input_str)
    print(ans)
'''






    



