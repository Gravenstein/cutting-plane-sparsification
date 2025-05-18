#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import util
import os
from typing import List
from tqdm import tqdm

'''
Important keys:
SolvingTime
Status
ProblemName
SparseTime
LP_Iter/sec_dualLP
NumSelectedCuts
DualIntegral
Nodes
'''

hline = '\\hline\n'


def bold(s: str):
    return f'\\textbf{{{s}}}'


def mono(s: str):
    return f'\\texttt{{{s}}}'


def multicolumn(width: int, s: str):
    return f'\\multicolumn{{{width}}}{{c}}{{{s}}}'


def print_table_row(args):
    print(' & '.join(args) + ' \\\\')


def shgeo(series, shift=0.0):
    return np.exp(1.0 / len(series) * np.sum(np.log(series + shift))) - shift


class DensityRangeFilter:

    def __init__(self, min_density: float, max_density: float, density_file: str):
        self.valid_problems = []
        for problem, density in util.read_csv_column(density_file, 'maxdensity').items():
            density = float(density)
            if density > min_density and density <= max_density:
                self.valid_problems.append(problem)

    def filter(self, data):
        include = np.zeros(len(data), dtype=bool)
        for i in range(len(data)):
            name: str = data['ProblemName'][i]
            name = name[:name.rfind('_')]
            if name in self.valid_problems:
                include[i] = True
        return include


class AtLeastOneCutFilter:

    def __init__(self, density_file: str):
        self.valid_problems = []
        for problem, density in util.read_csv_column(density_file, 'maxdensity').items():
            density = float(density)
            if density > 0.0:
                self.valid_problems.append(problem)

    def filter(self, data):
        include = np.zeros(len(data), dtype=bool)
        for i in range(len(data)):
            name: str = data['ProblemName'][i]
            name = name[:name.rfind('_')]
            if name in self.valid_problems:
                include[i] = True
        return include


class DefaultNotErrorFilter:

    def __init__(self, default_file: str):
        default = pd.read_csv(default_file)
        default.sort_values(inplace=True, by='ProblemName')
        default.reset_index(inplace=True)
        self.mask = default['Status'] != 'fail_abort'

    def filter(self, data):
        return self.mask


class Result:

    def __init__(self, file_name: str, min_density: int, normalizable: bool = True):
        self.data = pd.read_csv(file_name)
        self.data.sort_values(inplace=True, by='ProblemName')
        self.data.reset_index(inplace=True)

        self.name = os.path.basename(file_name).split('.')[0]
        self.normalizable = normalizable
        self.min_density = min_density

    def get_data(self, filters):
        include = np.ones(len(self.data), dtype=bool)
        for f in filters:
            include = include & f.filter(self.data)
        return self.data[include]

    def nsolved(self, filters):
        return sum(self.get_data(filters)['Status'] == 'ok')

    def ntimelimit(self, filters):
        return sum(self.get_data(filters)['Status'] == 'timelimit')

    def nerror(self, filters):
        return sum(self.get_data(filters)['Status'] == 'fail_abort')

    def ncuts(self, filters):
        return shgeo(self.get_data(filters)['NumSelectedCuts'], shift=1.0)

    def nnz(self, filters):
        return shgeo(self.get_data(filters)['CutNonzeros'], shift=1.0)

    def solve_time(self, filters):
        return shgeo(self.get_data(filters)['SolvingTime'], shift=1.0)

    def lp_iterations(self, filters):
        return shgeo(self.get_data(filters)['LP_Iterations_dualLP'], shift=100.0)

    def lp_iterations_per_second(self, filters):
        return shgeo(self.get_data(filters)['LP_Iter/sec_dualLP'], shift=10.0)

    def nodes(self, filters):
        return shgeo(self.get_data(filters)['Nodes'], shift=10.0)

    def nodes_per_second(self, filters):
        return shgeo(self.get_data(filters)['Nodes'] / self.get_data(filters)['SolvingTime'], shift=1.0)

    def lp_iterations_per_node(self, filters):
        return shgeo(self.get_data(filters)['LP_Iterations_dualLP'] / self.get_data(filters)['Nodes'], shift=10.0)


class Metric:
    def __init__(self, name: str, data_func, normalizable, best_func, default: Result):
        self.name = name
        self.data_func = data_func
        self.best_func = best_func
        self.normalizable = normalizable
        self.default = default

    def extract_value(self, result: Result, filters):
        value = self.data_func(result, filters)
        return value

    def find_best(self, values):
        return self.best_func(values) if self.best_func is not None else None


def print_density_separated_table(args, results: List[Result], metrics: List[Metric], partitions: List[int], global_filters=[], caption: str = '', label: str = ''):
    partition_width = 2*len(partitions)
    ncols = 3 + partition_width

    data = {}
    for result in tqdm(results, desc='Extracting data'):
        data[result.name] = {}
        for metric in metrics:
            data[result.name][metric.name] = [
                metric.extract_value(result, global_filters)]
            for i in range(1, len(partitions)):
                part_filter = DensityRangeFilter(
                    partitions[i-1] / 100, partitions[i] / 100, args.maxdensity_file)
                data[result.name][metric.name].append(
                    metric.extract_value(result, global_filters + [part_filter]))

    best = {}
    for metric in metrics:
        best[metric.name] = []
        for i in range(len(partitions)):
            values = [data[result.name][metric.name][i] for result in results]
            best[metric.name].append(metric.find_best(values))

    print('\\begin{table}[htbp]')
    print('\\centering')
    print(
        f'\\begin{{tabularx}}{{\\textwidth}}{{ccc*{{{partition_width}}}{{X}}}}')
    print_table_row([''] * ncols)
    print_table_row([multicolumn(3, ''), multicolumn(14, 'Max cut density')])
    print_table_row([bold('Config'), bold('Metric'), bold(
        'All')] + list(map(lambda x: multicolumn(2, f'{x}\\%'), partitions)))
    print(hline)

    for result in results:
        config = result.name
        first_metric = True
        for metric in metrics:
            values = data[config][metric.name]
            row = [mono(config) if first_metric else '',
                   metric.name]
            first_value = True
            for i, value in enumerate(values):
                if result.normalizable and metric.normalizable:
                    normalized = value / data[results[0].name][metric.name][i]
                    formatted = f'{normalized:.2f}'
                else:
                    formatted = f'{value:.0f}'
                if value == best[metric.name][i]:
                    formatted = bold(formatted)
                if i > 0 and partitions[i] <= result.min_density:
                    formatted = ''
                row.append(multicolumn(1 if first_value else 2, formatted))
                if first_value:
                    row.append(multicolumn(1, ''))
                    first_value = False
            row.append(multicolumn(1, ''))
            print_table_row(row)
            first_metric = False
        print(hline)

    print('\\end{tabularx}')
    if caption:
        print(f'\\caption{{{caption}}}')
    if label:
        print(f'\\label{{{label}}}')
    print('\\end{table}')


def __run(args):
    default = Result(args.default_file, 0, normalizable=False)

    results = [
        Result(util.parse_file('results/default.csv'), 0, normalizable=False),
        Result(util.parse_file('results/reject-05.csv'), 5),
        Result(util.parse_file('results/reject-10.csv'), 10),
        Result(util.parse_file('results/reject-20.csv'), 20),
        Result(util.parse_file('results/reject-40.csv'), 40),
        Result(util.parse_file('results/reject-80.csv'), 80),
        Result(util.parse_file('results/sparse-00.csv'), 0),
        Result(util.parse_file('results/sparse-05.csv'), 5),
        Result(util.parse_file('results/sparse-10.csv'), 10),
        Result(util.parse_file('results/sparse-20.csv'), 20),
        Result(util.parse_file('results/sparse-40.csv'), 40),
        Result(util.parse_file('results/sparse-80.csv'), 80)
    ]

    metrics = []
    if args.nsolved:
        metrics.append(Metric('nsolved', lambda data,
                       filters: data.nsolved(filters), False, max, default))
    if args.ntimelimit:
        metrics.append(Metric('ntimelimit', lambda data,
                       filters: data.ntimelimit(filters), False, min, default))
    if args.nerror:
        metrics.append(Metric('nerror', lambda data,
                       filters: data.nerror(filters), False, min, default))
    if args.time:
        metrics.append(Metric('time', lambda data, filters: data.solve_time(filters),
                              True, min, default))
    if args.it:
        metrics.append(Metric('it', lambda data, filters: data.lp_iterations(
            filters), True, min, default))
    if args.itps:
        metrics.append(Metric('it/s', lambda data,
                       filters: data.lp_iterations_per_second(filters), True, max, default))
    if args.nodes:
        metrics.append(Metric('nodes', lambda data,
                       filters: data.nodes(filters), True, min, default))
    if args.nodesps:
        metrics.append(Metric('nodes/s', lambda data,
                       filters: data.nodes_per_second(filters), True, max, default))
    if args.itpn:
        metrics.append(Metric('it/node', lambda data,
                       filters: data.lp_iterations_per_node(filters), True, min, default))
    if args.ncuts:
        metrics.append(Metric('ncuts', lambda data,
                       filters: data.ncuts(filters), True, None, default))
    if args.nnz:
        metrics.append(Metric('nnz', lambda data,
                       filters: data.nnz(filters), True, min, default))

    global_filters = []
    global_filters.append(AtLeastOneCutFilter(args.maxdensity_file))
    partitions = [0, 5, 10, 20, 40, 80, 100]

    print_density_separated_table(
        args, results, metrics, partitions, global_filters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxdensity_file', type=util.parse_file, default='data/maxdensity.csv',
                        help='CSV file containing the density of the densest cut for each problem.')
    util.add_boolean_flag(parser, 'exclude_default_error', default=False,
                          help='Exclude problems where the default config resulted in an error.')
    parser.add_argument('--default_file', type=util.parse_file, default='results/default.csv',
                        help='Experiment results with default config.')
    util.add_boolean_flag(parser, 'nsolved', default=False)
    util.add_boolean_flag(parser, 'ntimelimit', default=False)
    util.add_boolean_flag(parser, 'nerror', default=False)
    util.add_boolean_flag(parser, 'time', default=False)
    util.add_boolean_flag(parser, 'it', default=False)
    util.add_boolean_flag(parser, 'itps', default=False)
    util.add_boolean_flag(parser, 'nodes', default=False)
    util.add_boolean_flag(parser, 'nodesps', default=False)
    util.add_boolean_flag(parser, 'itpn', default=False)
    util.add_boolean_flag(parser, 'ncuts', default=False)
    util.add_boolean_flag(parser, 'nnz', default=False)

    args = parser.parse_args()
    __run(args)
