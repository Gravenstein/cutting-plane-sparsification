#!/usr/bin/python3

import argparse
import collections
import logging
import os
import statistics
from typing import List

LOG_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)-5.5s]  %(message)s')


def assert_file_exists(file_path: str) -> None:
    assert os.path.abspath(
        file_path) == file_path, f'{file_path} is a relative path'
    assert os.path.isfile(file_path), f'{file_path} is not a file'


def parse_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise ValueError(f'{file_path} does not exist')
    if not os.path.isfile(file_path):
        raise ValueError(f'{file_path} is not a file')
    return os.path.abspath(file_path)


def assert_dir_exists(dir_path: str) -> None:
    assert os.path.abspath(
        dir_path) == dir_path, f'{dir_path} is a relative path'
    assert os.path.isdir(dir_path), f'{dir_path} is not a directory'


def parse_dir(dir: str) -> str:
    if not os.path.exists(dir):
        raise ValueError(f'{dir} does not exist')
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not a directory')
    return os.path.abspath(dir)


def find_file(basename: str, directories: List[str], extensions: List[str]) -> str:
    assert directories
    assert extensions

    for dir in directories:
        for ext in extensions:
            filename = basename
            if ext:
                filename = f'{filename}.{ext}'
            filename = os.path.join(dir, filename)
            if os.path.isfile(filename):
                return filename
    assert_file_exists(filename)
    return filename


def add_boolean_flag(parser: argparse.ArgumentParser, name: str, **kwargs):
    parser.add_argument(f'--{name}', action='store_true', **kwargs)
    parser.add_argument(f'--no-{name}', dest=f'{name}', action='store_false')


def strip_seed_from_name(name: str) -> str:
    seed_start = name.rfind('_')
    assert seed_start > 0, f'First _ in position {seed_start} of {name}'
    try:
        seed = int(name[seed_start+1:])
    except ValueError as e:
        assert False, f'{name} does not end with a valid seed.'
    return name[:seed_start]


def read_csv_column(csv_file: str, column_name: str):
    with open(csv_file, 'r') as f:
        header = f.readline().strip().split(',')
        column_index = header.index(column_name)
        res = {}
        for i, line in enumerate(f.readlines()):
            line = line.strip().split(',')
            assert len(line) == len(
                header), f'Line {i} of {csv_file} has length {len(line)}, but header was {len(header)}.'
            assert line[0] not in res, f'Duplicate entries for {line[0]} in {csv_file}.'
            res[line[0]] = line[column_index]
    return res


def __extract_nnz(args):
    import pandas as pd

    ipet_data = pd.read_csv(args.ipet_csv)
    raw_nnz = collections.defaultdict(list)
    for _, row in ipet_data.iterrows():
        name = strip_seed_from_name(row['ProblemName'])
        raw_nnz[name].append(row['CutNonzeros'])
    csv_content = []
    for name, nnzs in raw_nnz.items():
        if args.seed_aggregator == 'min':
            aggr_nnz = min(nnzs)
        elif args.seed_aggregator == 'max':
            aggr_nnz = max(nnzs)
        elif args.seed_aggregator == 'mean':
            aggr_nnz = statistics.mean(nnzs)
        elif args.seed_aggregator == 'median':
            aggr_nnz = statistics.median(nnzs)
        csv_content.append((name, aggr_nnz))
    csv_content.sort(key=lambda tup: tup[0])
    print('problem,nnz')
    for name, nnz in csv_content:
        print(f'{name},{nnz}')


def __extract_densest_cut(args):
    with open(args.test_file, 'r') as f:
        print('name,maxdensity')
        for name in f:
            name = name.strip()
            densities = []
            for seed in args.seed:
                cut_stats_path = os.path.join(
                    args.result_dir, f'{name}_{seed}', 'original_cut_stats.csv')
                densities += read_csv_column(cut_stats_path,
                                             'density [%]').values()
            densities = list(map(float, densities))
            max_density = max(densities) if densities else 0
            print(name, max_density / 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Parameters for extract_nnz
    nnz_parser = subparsers.add_parser(
        'extractnnz', help='Extract cut nnz from ipet csv to separate csv.')
    nnz_parser.set_defaults(func=__extract_nnz)
    nnz_parser.add_argument('--ipet_csv', type=parse_file, required=True,
                            help='The ipet CSV containing nnz information.')
    nnz_parser.add_argument('--seed_aggregator', type=str,
                            default='max', choices=['min', 'max', 'mean', 'median'], help='How should separate seeds for the same instance be aggregated?')

    density_parser = subparsers.add_parser(
        'extractdensity', help="Extract maximum density of cuts generated for each problem instance (across seeds).")
    density_parser.set_defaults(func=__extract_densest_cut)
    density_parser.add_argument('--test_file', type=parse_file, required=True,
                                help='File containing problem names, one per line.')
    density_parser.add_argument('--result_dir', type=parse_dir, required=True,
                                help='Directory where results should be placed. Interpretation depends on subcommand.')
    density_parser.add_argument('-s', '--seed', type=int, action='append', required=True,
                                help='SCIP seeds, repeatable.')

    args = parser.parse_args()
    args.func(args)
