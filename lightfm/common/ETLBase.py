#-*- coding: utf-8 -*-
import os
import sys
import traceback
import gc
import abc

import pandas as pd
from tabulate import tabulate
import numpy as np
from pyflow import GraphBuilder
import pprint
from common.utils import Str2CodeAdaptor

_pp = pprint.PrettyPrinter(width=10, compact=True, indent=1)

def _get_print_result(result):
    if type(result) == pd.core.frame.DataFrame:
        print_result = tabulate(result.head(1).T, headers='keys', tablefmt='psql')
    elif any([
        type(result) == str,
        type(result) == list,
        type(result) == range,
        type(result) == tuple,
        type(result) == dict,
        type(result) == set,
        type(result) == frozenset,
        type(result) == int,
        type(result) == float,
        type(result) == complex,
        type(result) == bool,
        type(result) == bytes, 
        type(result) == bytearray,
        type(result) == memoryview, 
        type(result) == np.ndarray
    ]):
        print_result = _pp.pformat(result)
    else:
        print_result = str(type(result))
    if print_result.count('\n')>20:
        print_result = "\n".join(
            print_result.split('\n')[:8] + [
                '...'
            ] + print_result.split('\n')[-4:])
    return print_result
# * Component of Preprocess Base:
#
# [X] 1. function for setup configured variables (perhaps through a list of var string. Is it possible?)
# [X] 2. function for setup syntax sugar variables (such as SIX=6, TRUE = True)
# [X] 3. function for defining operators
# [X] 4. function for importing operators
# [X] 5. function for connecting operators
# [X] 6. function for defining the input and output variables through list of strings
# [X] 7. function for getting results
# [X] 8. function for defining the module_name
# [X] 9. build_collector (for building input and output collector)
# [ ] 10. allow connection to be
#       - [X] 10.1 manually set (with result_dir manually set, directory manually set.)
#       - [ ] or automatically set (with all result_dir saved in <tmp>/<module_name>/<tmp>/<output_name>)
# [X] 11. in __init__:
#   [X] 1. take pipe = None for deciding whether to connect to a existing pipe or initilize a new pipe
#   [X] 2. initilize: PARENT_TMP_DIR
#   [X] 3. define_operators
# [X] 12. in config:
#   [X] 1. build_collector (for input and output)
#   [X] 2. connect pipe
#
# Addon to ProcessBase:
# [ ] 1. Allow manual setting of temp file dir to the variables in collect_inputs/outputs
# [ ] 2. Setting color of collect block to purple.
# [ ] 3. Printing only the errors happended in the specialized module rather in the common module (for better user experinence)

class Setup:
    def __init__(self, **kargs):
        self.kargs = kargs

class ProcessBase():
    '''
    Setup steps:
    1. Inherent
    2. overwrite all methods
    3. .setup_vars
    4. .config
    5. get_result
    '''
    def __init__(self, required_process=None, save_tmp=True, experiment_name='', **kargs):
        self.name = self.module_name()
        if experiment_name == '':
            self.PARENT_TMP_DIR = f'data/{self.name}'
        else:
            self.PARENT_TMP_DIR = f'data/{experiment_name}/{self.name}'
        self._save_tmp = save_tmp
        self.options = kargs
        if required_process:
            self.pipe_config = required_process.pipe_config
            self.pipe = required_process.pipe
            self.pipe._save_tmp = save_tmp
        else:
            self.pipe_config = PipeConfigBuilder()
            self.pipe = PipelineBuilder(pipe=self.pipe_config, save=save_tmp)
        self._define_operators()
        self.var_set = False
        self.config_done = False
        self.build_input_collector = False


    @property
    @abc.abstractmethod
    def module_name(self):
        '''
        Example:

        return "module_name"
        '''
        return "preprocess"
    @abc.abstractmethod
    def fix_vars(self):
        '''
        Example:

        return Setup(
            SIX=6,
            TRUE=True
        )
        '''
        return Setup()

    @abc.abstractmethod
    def packages(self):
        '''
        Example:

        return [
            'dev.etl.load_data',
            'dev.etl.risk_management'
        ]
        '''
        return []
    @abc.abstractmethod
    def define_functions(self, pipe):
        '''
        Example:

        @pipe._func_
        def get_today(etl):
            return etl.today
        '''
        pass



    @abc.abstractmethod
    def inputs(self):
        '''
        Example:

        return ['input1', 'input2', 'input3']
        '''
        pass
    @abc.abstractmethod
    def connections(self, **kargs):
        '''
        return a list of paired tuples, in each of which
            the first element being the connection python code and
            the second element a list of strings the names of the temporary files of the outputs.

            The second element can also be None, if saving temporary file is unneccesary for the outputs,
                or a string if there is only one output in the connection python code.

        Example:

        return [
            ('w106 = get_w106(etl)', ['w106.feather']),
            ('w107 = get_w107(etl)', ['w107.feather'])
        ]
        '''
        pass

    @abc.abstractmethod
    def outputs(self):
        '''
        Example:

        return ['output1', 'output2', 'output3']
        '''
        pass
    # Public Methods:
    def setup_vars(self, **kargs):
        assert self.pipe_config
        fix_kargs = self.fix_vars().kargs
        if len(kargs)!=0:
            self.build_input_collector = True
        dynamic_kargs = kargs
        self.config_vars = {
            **dynamic_kargs,
            **fix_kargs
        }
        self.pipe_config.setups(**self.config_vars)
        self.var_set = True

    def config(self, verbose=True, **kargs):
        if not self.var_set:
            self.setup_vars(**kargs)

        if self.build_input_collector:
            self._build_collector(self.inputs(), mode = 'input', verbose=verbose)
        self._connect(self.pipe, verbose=verbose, **self.options)
        self.config_done = True

    def get_result(self, verbose=False, load_tmp = False):
        if not self.config_done:
            self.config(verbose=verbose)
        result_list = []
        for output_var_str in self.outputs():
            result_list.append(getattr(self.pipe, output_var_str).get(
                verbose=verbose, load_tmp = load_tmp))
        return tuple(result_list)

    # Private Methods:
    def _connect(self, pipe, verbose=True, **kargs):
        for item in self.connections(**kargs):
            if isinstance(item, tuple):
                if len(item) == 2:
                    connection, output_files = item
                elif len(item) == 1:
                    connection, output_files = item[0], None
                elif len(item) == 0:
                    continue
            elif isinstance(item, str):
                connection, output_files = item, None
            else:
                connection, output_files = item, None
            try:
                if output_files == None:
                    pass
                else:
                    if isinstance(output_files, str):
                        output_files = [output_files]
                    output_files = [f'{self.PARENT_TMP_DIR}/tmp/{ofile}' for ofile in output_files]
                if isinstance(connection, str):
                    pipe.setup_connection(
                        connection,
                        result_dir=output_files, 
                        verbose=verbose
                        )
                else:
                    # conn = connection(required_process=self, save_tmp = self._save_tmp) # CustItemRecmd
                    connection.config(verbose=verbose)
            except:
                traceback_info = ''.join(
                traceback.format_exception(
                    *sys.exc_info()))
                raise ValueError(f'Error in {item}: {traceback_info}')

    def _define_operators(self):
        assert self.pipe
        packages = self.packages()
        print('packages in _define_operators:', packages) 
        self._define_operators_by_import(self.pipe, packages)
        self.define_functions(self.pipe)

    def _define_operators_by_import(self, pipe, packages):
        if type(pipe.func_source) == list and type(packages) == list:
            pipe.func_source = list(set(pipe.func_source) | set(packages))
        elif type(pipe.func_source) == list and type(packages) == str:
            pipe.func_source.append(packages) 
        elif type(pipe.func_source) == str and type(packages) == list: 
            pipe.func_source = [pipe.func_source] + packages
        elif type(pipe.func_source) == str and type(packages) == str: 
            pipe.func_source = [pipe.func_source, packages]
        elif pipe.func_source==None:
            pipe.func_source = packages

    def _build_collector(self, _vars, mode, verbose=True):
        assert mode == 'input' or mode == 'output'
        assert type(_vars) == list
        TMP_DIR = self.PARENT_TMP_DIR + f'/{mode}s'
        FUNC_NAME = f'collect_{mode}s_for_{self.name}'

        @self.pipe._func_
        def gather(*args, **kargs):
            return list(args) + [kargs[k] for k in kargs.keys()]

        exec(f'self.pipe._{FUNC_NAME} = self.pipe._gather')

        def generate_arg_str(var):
            if var in self.config_vars:
                return f'{var}={var}'
            else:
                return var
        code_str = f'{",".join(_vars)} = {FUNC_NAME}({",".join([generate_arg_str(v) for v in _vars])})'
        self.pipe.setup_connection(
            code_str, 
            verbose=verbose, 
            color='darkseagreen2'# search for color name here: https://graphviz.org/doc/info/colors.html
            # result_dir = [f'{TMP_DIR}/{var}' for var in _vars]
        )


class PipeConfigBuilder:
    def __init__(self):
        self.pyflow_GB = GraphBuilder()
        self.graph_dict = dict()

    def add(self, var_name, value, rank=None, color='gray', shape='cylinder', fontsize=None):
        def current_process():
            return value
        current_process_name = f'{var_name}={value}'
        # method_alias = current_process_name
        n_out = 1

        self.pyflow_GB.add(current_process,
                           method_alias = var_name,
                           output_alias = _get_print_result(value),
                           n_out = n_out,
                           rank = rank,
                           color = color,
                           shape = shape,
                           fontsize = fontsize
                          )
        # no call function here, directly call the method in add.
        pf_output = self.pyflow_GB()
        config_module = DataNode(current_process_name, [])
        config_module.set_process(current_process)
        config_module.set_n_out(n_out)
        config_module.set_pf_output_node(pf_output)
        self.graph_dict[pf_output.get_node_uid()] = config_module
        return config_module

    def view(self, *args, **kargs):
        return self.pyflow_GB.view(
            *args,
            **kargs
        )
    def view_dependency(self, *args, **kargs):
        return self.pyflow_GB.view_dependency(*args, **kargs)

    def setups(self, env = None, **kargs):
        for var_name, value in kargs.items():
            config_module = self.add(var_name, value)
            if env: # not None, can be globals() or an object.
                Str2CodeAdaptor.add_var(env, var_name, config_module)
            else:
                Str2CodeAdaptor.add_var(self, var_name, config_module)

class PipelineBuilder():
    # - [ ] load the function only if the function does not exist
    # - [ ] load the function in the beginning if source_func is provided
    def __init__(self, pipe=None, func_source=None, load_tmp = True, save=True):
        '''
        Input:
            - pipe: the pre-request pipe or a config object from PipeConfigBuilder.
            - func_source: the name of the module/package holding the functions used in this pipeline, a list of such names, or
                the globals() dictionary if the functions is defined in main function.
        '''
        self.current_process = None
        self.current_process_name = None
        self.pipe = pipe
        self.func_source = func_source
        if pipe:
            self.pyflow_GB = pipe.pyflow_GB
        else:
            self.pyflow_GB = GraphBuilder()

        self.graph_dict = dict()
        self.data_info_holder = _DataInfoHolder()
        self._current_mode = 'arg_only'
        self._load_tmp = load_tmp
        self.save = save

    def add(self, process, method_alias=None, output_alias=None, result_dir=None,
            n_out=1, rank=None, color='lightblue', shape=None, fontsize=None):
        self.current_process = process
        self.current_process_name = process.__name__
        self.n_out = n_out

        if type(result_dir) == list:
            assert len(result_dir) == n_out
            for out, path in zip(output_alias, result_dir):
                assert out == path.split("/")[-1].split(".")[0]
        if result_dir:
            color = 'pink'
        if self._load_tmp:
            self.result_dir = result_dir
        else:
            self.result_dir = None

        self.pyflow_GB.add(process,
                           method_alias = method_alias,
                           output_alias = output_alias,
                           n_out = n_out,
                           rank = rank,
                           color = color,
                           shape = shape,
                           fontsize = fontsize
                          )
        return self
    def __call__(self, *args, **kwargs):
        assert self.current_process
        assert self.current_process_name
        # print("In __call__, args, kargs:", args, kwargs)
        pf_input = [arg.pf_output_node for arg in args]

        pf_kwargs = dict([(key, self._to_pf_out_node(arg)) for key, arg in kwargs.items()])

        pf_output = self.pyflow_GB(*pf_input, **pf_kwargs)

        if self.result_dir != None and self.save == True:
            save = True
        else:
            save = False

        process_module = DataNode(
            self.current_process_name,
            list(args),
            result_dir = self.result_dir,
            save = save,
            **kwargs
        )

        process_module.set_process(self.current_process)
        process_module.set_n_out(self.n_out)

        if self.n_out > 1:

            outs = []
            for i, pf_output_node in zip(range(self.n_out), pf_output):

                out = SelectResult(
                self.current_process_name + f'[{i}]',
                [process_module],
                selected_indices=[i],
                save=self.save
                )
                out.set_pf_output_node(pf_output_node)
                self.graph_dict[pf_output_node.get_node_uid()] = out
                outs.append(out)

            return outs

        else:
            process_module.set_pf_output_node(pf_output)
            self.graph_dict[pf_output.get_node_uid()] = process_module
            return process_module
    def _to_pf_out_node(self, arg):
        if type(arg).__name__ == 'DataNode+ETLBase':
            return arg.pf_output_node
        if type(arg).__name__ == 'SelectResult+ETLBase':
            return arg.pf_output_node
        return arg
    def view(self, *args, option = 'arg_only', **kargs):
        if ('summary' in kargs) and (kargs['summary'] == False):
            self._assign_datanode_info(option = option)
        return self.pyflow_GB.view(
            *args,
            **kargs
        )
    def view_dependency(self, *args, option = 'arg_only', **kargs):
        if ('summary' in kargs) and (kargs['summary'] == False):
            self._assign_datanode_info(option = option)
        return self.pyflow_GB.view_dependency(*args, **kargs)

    def _assign_datanode_info(self, option = 'arg_only'):
        assert option == 'all' or option == 'light' or option == 'arg_only'
        if option == 'arg_only':
            for node_id, data_node in self.graph_dict.items():
                graph_node = self.pyflow_GB.graph_dict[node_id]
                graph_node['alias'] = "_".join(node_id.split("_")[:-1])
            return
        if self._current_mode == option:
            return
        else:
            if option == 'all' or option == 'light':
                for node_id, data_node in self.graph_dict.items():
                    graph_node = self.pyflow_GB.graph_dict[node_id]
                    graph_node['alias'] = self._obtain_datanode_info(node_id, data_node, option = option)
                    graph_node['attributes']['fontsize']=1
    def _obtain_datanode_info(self, node_id, data_node, option = 'light'):
        assert option == 'all' or option == 'light'
        if option == 'all':
            data_name = "_".join(node_id.split("_")[:-1])
            _size_ = self.data_info_holder.get_size(node_id, data_node)
            _type_ = self.data_info_holder.get_type(node_id, data_node)
            _data_content_ = self.data_info_holder.get_data_content(node_id, data_node)
            result_string = f'{data_name}\ntype:{_type_}\nsize: {_size_} bytes\n{_data_content_}'
        else:
            data_name = "_".join(node_id.split("_")[:-1])
            _size_ = self.data_info_holder.get_size(node_id, data_node)
            _type_ = self.data_info_holder.get_type(node_id, data_node)
            result_string = f'{data_name}\ntype:{_type_}\nsize: {_size_} bytes'
        self.data_info_holder.clean_up_result()
        return result_string

    def is_leaf(self, arg_name):
        info = self.pyflow_GB.graph_dict[arg_name]
        if 'type' not in info or 'children' not in info:
            return False
        else:
            return info['type'] == 'data' and len(info['children']) == 0

    def get_all_ancestor_pf_data_nodes(self, graph_dict, node):
        data_nodes = dict()
        def recursive(node):
            try:
                node_id = node.pf_output_node.get_node_uid()
                data_nodes[node_id] = (graph_dict[node_id], node)
            except:
                pass
            finally:
                for parent_node in node.pre_request_etls:
                    recursive(parent_node)
        recursive(node)
        return data_nodes

    def _func_(self, _func):
        # decorator for adding a function as a member of the class.
        exec(f'self._{_func.__name__} = _func')
        return _func

    def _rep_func_(self, func_name):
        '''
        Inputs:
            - func_name: the name of the function that should be replace
            - new_func: the new function (type: method)
        '''
        matched_operations = self._select_operation_by_name(func_name)
        output_nodes_list = self._get_output_nodes_list(matched_operations)
        def decorator(new_func):
            self._assign_new_function(output_nodes_list, new_func)
        return decorator

    def _select_operation_by_name(self, func_name):
        '''
        Input:
            - func_name: the name of the function
        Output:
            - matched_operations: a list of operation nodes in pyflow graph_dict
        '''
        def is_selected_operation(node_id, func_name):
            node = self.pyflow_GB.graph_dict[node_id]
            if 'type' in node and node['type'] == 'operation' and node['alias'] == func_name:
                return True
            else:
                return False
        matched_operations = list(filter(
            lambda node_id: is_selected_operation(node_id, func_name),
            self.pyflow_GB.graph_dict.keys()
        ))
        return matched_operations
    def _get_output_nodes_list(self, operations):
        '''
        Inputs:
            - matched_operations: a list of operation nodes in pyflow graph_dict.
        Outputs:
            - output_nodes_list: a list of children list where each element is a output data node
                of an operation.
        '''
        output_nodes_list = [self.pyflow_GB.graph_dict[operation]['children'] for operation in operations]
        return output_nodes_list

    def _assign_new_function(self, output_nodes_list, new_func):
        '''
        Given a list of output nodes where the new function should be assigned as their process.
        Note: the new function is injected from one of the output nodes of the operation.
        '''
        for output_nodes in output_nodes_list:
            if len(output_nodes) > 1:
                assert type(self.graph_dict[output_nodes[0]]).__name__ == 'SelectResult+ETLBase'
                self.graph_dict[output_nodes[0]].pre_request_etls[0].set_process(new_func)
            else:
                assert type(self.graph_dict[output_nodes[0]]).__name__.split('+')[0] == 'DataNode'
                self.graph_dict[output_nodes[0]].set_process(new_func)

    def _set_func_str_as_private_functions(self, func_str, func, func_source, verbose=True):
        '''
        Inputs: 
            - func: the func assign during connection.
            - func_source: the imported function containing the functions.
            - func_str: the function string extracted from the connection 
                code string. 
        '''
        private_func_str = f"self._{func_str.replace('.','_')}"

        if func:
            exec(self._print_exec(
                f"{private_func_str} = func", 
                verbose=verbose))
        else: 
            # If no custom function
            # insert function from global
            try:
                eval(private_func_str) 
                # if the function cannot be found, 
                # import packages to load the function. 
            except:
                if type(func_source) == str:
                    # only one package imported 
                    if '.' in func_str:
                        func_str = func_str.split(".")[0]
                    exec(self._print_exec(
                        f'from {func_source} import {func_str}', 
                        verbose=verbose))
                elif type(func_source) == list:
                    # multiple packages imported 
                    for s in func_source:
                        exec(self._print_exec(
                            f'from {s} import *', 
                            verbose=verbose))
                else:
                    raise ValueError(
                        f'{str(func_source)} is not string or list of strings')
                exec(self._print_exec(
                    f"{private_func_str} = {func_str}", 
                    verbose=verbose))
        return private_func_str
    def _print_exec(self, code_str, verbose=False):
        if verbose:
            print('    ', code_str)
        return code_str
        
    def setup_connection(self, code_str, env = None, result_dir = None, func = None, verbose=False, color='lightblue'):
        '''
        This function allows self.add to be simplify with simple python function call command.
        Input:
            - obj: determine where to put the output variables of the function call.
                - if `obj` == globals(): put into global environment.
                - if `obj` == None: put into self
                - otherwise: put into the object represented by `obj`
        '''
        if env: # not None, can be globals() or an object.
            pass
        else:
            env = self
        fixed_code = Str2CodeAdaptor.fix_code(code_str)
        if verbose:
            if result_dir:
                print('[Connect]', fixed_code, '[Result Dir]', result_dir)
            else:
                print('[Connect]', fixed_code)
        
        out_vars, func_str, args_str_list, kwargs_str_dict = Str2CodeAdaptor.breakdown_function_call(fixed_code)
        private_func_str = self._set_func_str_as_private_functions(
            func_str, func, self.func_source, verbose=False)
        
        func_added_pipe = self.add(
            eval(private_func_str),
            method_alias = func_str,
            color = color, 
            n_out = len(out_vars),
            output_alias = out_vars,
            result_dir = result_dir
        )
        # print('func_added_pipe:', func_added_pipe)
        # TODO: evaluate values in input_dict :
        #  [V] if env == globals(), get global_variable from global.
        #  [V] if env == None, use self as the prefix of object values.
        #  [V] if env == an object, use this object as the prefix (meaning that the pre-request objects are from this object.)

        args = [Str2CodeAdaptor.get_var(env, var) for var in args_str_list]

        def force_eval(value_str):
            try:
                try:
                    return eval(value_str)
                except:
                    return Str2CodeAdaptor.get_var(env, value_str)
            except:
                if type(env) != dict:
                    # if cannot find in this pipe, how about try to find it from the requested pipe connect to this pipe (i.e., env.pipe)
                    return Str2CodeAdaptor.get_var(env.pipe, value_str)
        kargs = dict([(var, force_eval(value_str)) for var, value_str in kwargs_str_dict.items()])
        # print("In setup connection, args, kargs:",args, kargs)
        out_nodes = func_added_pipe(*args, **kargs)

        if len(out_vars) > 1:
            for out_name, node in zip(out_vars, out_nodes):
                Str2CodeAdaptor.add_var(env, out_name, node)
        else:
            Str2CodeAdaptor.add_var(env, out_vars[0], out_nodes)

class _DataInfoHolder:
    def __init__(self):
        self.print_content_dict = dict()
        self.size_dict = dict()
        self.type_dict = dict()
        self._result = None
    def get_size(self, node_id, data_node):
        if node_id in self.size_dict:
            return self.size_dict[node_id]
        else:
            if self._result is not None:
                self.size_dict[node_id] = sys.getsizeof(self._result)
            else:
                self._result = data_node.get()
                self.size_dict[node_id] = sys.getsizeof(self._result)
            return self.size_dict[node_id]
    def get_type(self, node_id, data_node):
        if node_id in self.type_dict:
            return self.type_dict[node_id]
        else:
            if self._result is not None:
                self.type_dict[node_id] = type(self._result)
            else:
                self._result = data_node.get()
                self.type_dict[node_id] = type(self._result)
            return self.type_dict[node_id]
    def get_data_content(self, node_id, data_node):
        if node_id in self.print_content_dict:
            return self.print_content_dict[node_id]
        else:
            if self._result is not None:
                self.print_content_dict[node_id] = _get_print_result(self._result)
            else:
                self._result = data_node.get()
                self.print_content_dict[node_id] = _get_print_result(self._result)
            return self.print_content_dict[node_id]
    
    def clean_up_result(self):
        self._result = None
        gc.collect()

class ETLBase:
    def __init__(self, process_name, pre_request_etls=[], save=False, in_memory = True):
        self.process_name = process_name
        self.pre_request_etls = pre_request_etls
        for etl_obj in self.pre_request_etls:
            etl_obj.add_child(self)
        self.save = save
        self._in_memory = in_memory
        self._in_memory_results = None
        self._children = []
        self._processed = False

    def is_processed(self):
        return self._processed

    def add_child(self, child):
        self._children.append(child)

    def all_children_processed(self):
        return all([child.is_processed() for child in self._children])

    def remove_in_memory_results(self, verbose=False):
        self._in_memory_results = None
        gc.collect()
        if verbose:
            print(f'[REMOVE] result of "{self.process_name}"')

    def run(self, verbose=False, load_tmp = True, handle_wastes = True):
        '''
        Check if the pre-request results are completed.
        If completed, load the result. Otherwise, run the pre-request ETL process.
        '''
        inputs = []
        for pre_etl in self.pre_request_etls:
            if pre_etl.is_complete():
                if verbose:
                    print(f'[LOAD] result of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.load_result(
                    verbose=verbose,
                    load_tmp=load_tmp))
            else:
                #if verbose:
                    #print(f'[IN] process of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.run(
                    verbose=verbose,
                    load_tmp=load_tmp,
                    handle_wastes=handle_wastes
                ))
        if verbose:
            print(f'[RUN] process of "{self.process_name}"')
        results = self.process(inputs)
        self._processed = True
        if verbose:
            print(f'[COMPLETE] {self.process_name}')
        if handle_wastes:
            for pre_etl in self.pre_request_etls:
                if pre_etl.all_children_processed():
                    pre_etl.remove_in_memory_results(verbose=verbose)
        del inputs
        gc.collect()


        self.save_result(results, verbose=verbose, save = self.save)
        return results

    def is_complete(self):
        '''
        This function check if the temporary result file is already saved, to notify
        whether the process should be triggered before the next ETL is processed.
        '''
        if self._in_memory:
            if self._in_memory_results == None:
                return False
            else:
                return True
        else:
            return False

    def load_result(self, verbose=False, load_tmp = True):
        '''
        This function load the temporary result file saved by "save_result" function.
        Should be override if save_result is override.
        '''
        if self._in_memory:
            return self._in_memory_results
        else:
            pass

    def process(self, inputs):
        '''
        This is the main process and should be override.
        input:
         - inputs: a list containing the inputs
        output:
         - outputs: a list containing the outputs
        '''
        outputs = inputs
        return outputs

    def save_result(self, results, verbose=False, save=True):
        '''
        Save result for the next ETL Process.
        Sould be considered overrided if re-use of processed data is considered
        '''
        if self._in_memory:
            self._in_memory_results = results
        else:
            pass


class ETLwithDifferentResults(ETLBase):
    def __init__(self, process_name, pre_request_etls, result_dir, save=True, in_memory=True):
        super(ETLwithDifferentResults, self).__init__(
            process_name,
            pre_request_etls=pre_request_etls,
            save=save
        )
        if type(result_dir) == list:
            self.result_dirs = result_dir
        else:
            self.result_dirs = [result_dir]
        # self._create_data_folder()
        self._in_memory = in_memory
        self._in_memory_results = None
    def _create_data_folder(self, verbose=True):
        result_dirs = self.result_dirs
        # create folders on initialization
        for result_path in result_dirs:
            if ".." in result_path:
                break # Note: do not create folder outside current directory
            folder_path = ""
            for folder_name in result_path.split('/')[:-1]:
                folder_path += f"{folder_name}/"
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                    if verbose:
                        print(f"You have created directory: {folder_path}")

    def is_complete(self):
        ans = False
        if self._in_memory:
            if isinstance(self._in_memory_results, list):
                ans = True
        if ans:
            return True
        else:
            ans = all([os.path.exists(file_dir) for file_dir in self.result_dirs])
            return ans

    def save_result(self, results, verbose=False, save = True):
        self._create_data_folder(verbose=verbose)
        if save:
            if verbose:
                print(f'[SAVE] result of "{self.process_name}"')
            for i, file_dir in enumerate(self.result_dirs):
                # feather.write_dataframe(results[0], self.result_dir)
                if '.h5' in file_dir:
                    results[i].to_hdf(file_dir, key=file_dir.split('.')[0].split('/')[-1], mode='w')
                elif '.feather' in file_dir:
                    import feather
                    feather.write_dataframe(results[i].reset_index(), file_dir)
                elif '.npy' in file_dir:
                    np.save(file_dir.split('.npy')[0], results[i])
                if verbose:
                    print(f' as {file_dir}')
        if self._in_memory:
            self._in_memory_results = results
            if verbose:
                print(f'[SAVE] result of "{self.process_name}"')
                print('in memory')

    def load_result(self, verbose=False, load_tmp = True):
        if self._in_memory and self._in_memory_results is not None:
            if verbose:
                print(f' from memory')
            return self._in_memory_results
        if load_tmp:
            results = []
            for file_dir in self.result_dirs:
                if '.h5' in file_dir:
                    results.append(pd.read_hdf(file_dir, key=file_dir.split('.h5')[0].split('/')[-1], mode='r'))
                elif '.feather' in file_dir:
                    import feather
                    reverse_reset_index = lambda table: table.set_index(table.columns[0])
                    results.append(reverse_reset_index(feather.read_dataframe(file_dir)))
                elif '.npy' in file_dir:
                    # np.save(file_dir.split('.npy')[0], feature_mapper)
                    np_result = np.load(file_dir, allow_pickle=True)
                    try:
                        np_result = np_result.item()
                    except:
                        pass
                    results.append(np_result)
                if verbose:
                    print(f' from {file_dir}')
            return results




class ETLPro:
    def __new__(cls, process_name, pre_request_etls, result_dir=None, **kwargs):
        if result_dir:
            super_class = ETLwithDifferentResults
        else:
            super_class = ETLBase
        cls = type(cls.__name__ + '+' + super_class.__name__, (cls, super_class), {})
        return super(ETLPro, cls).__new__(cls)

    def __init__(self, process_name, pre_request_etls, result_dir=None, save=True, **kwargs):
        if result_dir:
            super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir, save=save)
        else:
            super(ETLPro, self).__init__(process_name, pre_request_etls=pre_request_etls, save=save)
    def set_pf_output_node(self, pf_output_node):
        self.pf_output_node = pf_output_node


class SelectResult(ETLPro):
    '''
    This ETL process selects particular results from previous ETLs.
    By default, it extract the first result.
    '''

    def __init__(self, process_name, pre_request_etls, selected_indices=[0], result_dir=None, save=True):
        super(SelectResult, self).__init__(process_name, pre_request_etls, result_dir=result_dir, save=save)
        self.selected_indices = selected_indices

    def process(self, inputs):
        assert len(self.selected_indices) < len(inputs)
        return [inputs[i] for i in self.selected_indices]
    def get(self, verbose=False, load_tmp=False, handle_wastes=True):
        assert len(self.selected_indices) == 1
        return self.run(
            verbose=verbose,
            load_tmp=load_tmp,
            handle_wastes=handle_wastes)[0]


class DataNode(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, save=True, **kwargs):
        super(DataNode, self).__init__(process_name, pre_request_etls, result_dir=result_dir, save=save)
        self.kwargs = dict([(key, self._get_real_var(arg)) for key, arg in kwargs.items()])
        self.n_out = 1

    def _get_real_var(self, arg):
        '''
        Some of the args in kwargs is not real data but ETLPro Objects.
        To solve the problem, convert them all to data using .get.
        '''
        if type(arg).__name__ == 'DataNode+ETLBase':
            return arg.get()
        if type(arg).__name__ == 'SelectResult+ETLBase':
            return arg.get()
        return arg

    def set_process(self, current_process):
        self.current_process = current_process

    def set_n_out(self, n_out):
        self.n_out = n_out

    def process(self, inputs):

        result = self.current_process(*inputs, **self.kwargs)
        if self.n_out == 1:
            return [result]
        else:
            return result

    def get(self, verbose=False, load_tmp=True, handle_wastes=True):
        assert self.n_out == 1
        return self.run(
            verbose=verbose,
            load_tmp=load_tmp,
            handle_wastes=handle_wastes)[0]

# TODO:
# - [V] change temp object name to obj rather than the function name
# - [V] make the ETLBase allow save when a result directory is assigned
# - [V] allow saving of multiple files (ETL might have multiple results)
# - [V] allow single input and single output (not list)
# - [V] allow subset selection from previous ETL process
# - [X] allow the input to be a dictionary and the output to be a dictionary, too
# - [V] make the ETL Process object allows two step construction like nn.module. 1. first initialized with configuration . 2. Be called to assign inputs and obtain outputs later
# - [ ] incorporate google drive download as the first step of ETL
# - [V] allows zero input ETL if the ETL does not have previous ETL
