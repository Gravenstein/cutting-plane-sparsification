#!/usr/bin/python3

import argparse
import os
import util
import subprocess
import sys

from typing import List


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class Problem:
    def __init__(self, name: str, status: str, nvar: str, nvar_bin: str, nvar_int: str, nvar_cont: str, ncons: str, nnz: str, group: str, obj: str, tags: str) -> None:
        self.name = name
        self.status = status
        self.nvar = int(nvar)
        self.nvar_bin = int(nvar_bin)
        self.nvar_int = int(nvar_int)
        self.nvar_cont = int(nvar_cont)
        self.ncons = int(ncons)
        self.nnz = int(nnz)
        self.group = group
        obj = obj.strip('*')
        try:
            self.obj = float(obj)
        except ValueError:
            self.obj = None
        self.tags = list(tags.split(' '))


class Condition:
    def __init__(self, desc: str) -> None:
        self.ntested = 0
        self.npassed = 0
        self.nfailed = 0
        self.desc = desc

    def problem_is_valid(self, problem: Problem) -> str:
        reason = self._is_valid_impl(problem)
        self.ntested += 1
        if len(reason) == 0:
            self.npassed += 1
        else:
            self.nfailed += 1
        return reason

    def _is_valid_impl(self, problem: Problem) -> str:
        raise NotImplementedError()


class StatusCondition(Condition):
    def __init__(self):
        super().__init__('Status is easy or hard')

    def _is_valid_impl(self, problem: Problem) -> str:
        if problem.status == 'easy' or problem.status == 'hard':
            return ''
        return f'Has status "{problem.status}"'


class ExcludeTagCondition(Condition):
    def __init__(self, tag) -> None:
        super().__init__(f'Does not have tag {tag}')
        self.tag = tag

    def _is_valid_impl(self, problem: Problem) -> str:
        return f'Has tag "{self.tag}"' if self.tag in problem.tags else ''


class NoObjectiveValueCondition(Condition):
    def __init__(self) -> None:
        super().__init__('Has a finite objective value')

    def _is_valid_impl(self, problem: Problem) -> str:
        return f'No valid objective value ({problem.obj})' if problem.obj is None else ''


class NoSolutionCondition(Condition):
    def __init__(self, solution_directory):
        super().__init__(f'Has an available solution')
        self.solution_directory = solution_directory

    def _is_valid_impl(self, problem: Problem) -> str:
        filename = f'{self.solution_directory}/{problem.name}.sol.gz'
        return f'No solution file: {filename}' if not os.path.isfile(filename) else ''


class NoCutsCondition(Condition):
    def __init__(self, max_density_file: str):
        super().__init__(f'Uses at least one cut')
        self.max_density = util.read_csv_column(max_density_file, 'maxdensity')

    def _is_valid_impl(self, problem: Problem) -> str:
        return 'No cuts were used' if problem.name in self.max_density and self.max_density[problem.name] == 0.0 else ''


class LoadableBySCIPCondition(Condition):
    def __init__(self, model_directory, solution_directory, scip_binary, scip_settings_file):
        super().__init__(f'SCIP accepts the solution')
        self.model_directory = model_directory
        self.solution_directory = solution_directory
        self.scip_binary = scip_binary
        self.scip_settings_file = scip_settings_file

    def _is_valid_impl(self, problem: Problem) -> str:
        modelfile = f'{self.model_directory}/{problem.name}.mps.gz'
        solfile = f'{self.solution_directory}/{problem.name}.sol.gz'
        util.assert_file_exists(modelfile)
        util.assert_file_exists(solfile)
        scip_result = subprocess.run([
            self.scip_binary,
            '-c', f'set load {self.scip_settings_file}',
            '-c', f'read {modelfile}',
            '-c', f'read {solfile}',
            '-c', 'transform',
            '-c', 'quit'
        ], stdout=subprocess.PIPE,  stderr=subprocess.STDOUT)

        output = scip_result.stdout.decode('utf-8')
        if '1/1 feasible solution given by solution candidate storage' not in output:
            return f'{solfile} was rejected by SCIP'
        if scip_result.returncode != 0:
            eprint(
                f'Reading {problem.name} returned {scip_result.returncode} != 0')
            eprint(output)
            raise RuntimeError()
        return ''


def extract_problem_data(problem_file: str) -> List[Problem]:
    util.assert_file_exists(problem_file)
    problems = []
    with open(problem_file, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f.readlines():
            line = line.strip().split(',')
            line = list(map(lambda s: s.strip(), line))
            assert len(line) == len(header)
            problems.append(Problem(*line))
    return problems


def create_dataset(args):
    conditions: List[Condition] = [
        StatusCondition(),
        ExcludeTagCondition('feasibility'),
        ExcludeTagCondition('numerics'),
        ExcludeTagCondition('infeasible'),
        ExcludeTagCondition('no_solution'),
        NoObjectiveValueCondition(),
        NoSolutionCondition(args.solution_directory),
        LoadableBySCIPCondition(args.model_directory,
                                args.solution_directory, args.scip_binary, args.scip_settings_file)
    ]
    if args.max_density_file:
        conditions.append(NoCutsCondition(args.max_density_file))

    problems = extract_problem_data(args.dataset_file)
    for problem in problems:
        for condition in conditions:
            reason = condition.problem_is_valid(problem)
            if len(reason) != 0:
                break
        if len(reason) == 0:
            if args.verbose:
                eprint(f'{problem.name} passed all conditions')
            print(problem.name)
        else:
            if args.verbose:
                eprint(f'Rejecting {problem.name}: {reason}')
    if args.verbose:
        eprint('Condition statistics:')
        desclength = max(len(condition.desc) for condition in conditions)
        for condition in conditions:
            eprint('{0:{1}}: {2:3} / {3:3}'.format(condition.desc,
                   desclength, condition.npassed, condition.ntested))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to create a dataset.')
    parser.add_argument('--dataset_file', type=util.parse_file, required=True,
                        help='File with dataset information.')
    parser.add_argument('--model_directory', type=util.parse_dir, required=True,
                        help='Directory where model files are found.')
    parser.add_argument('--solution_directory', type=util.parse_dir, required=True,
                        help='Directory where solution files are found.')
    parser.add_argument('--scip_binary', type=util.parse_file, required=True,
                        help='SCIP binary.')
    parser.add_argument('--scip_settings_file', type=util.parse_file, default='configs/3600-seconds-nosparse.set',
                        help='Settings to use when verifying using SCIP.')
    parser.add_argument('--max_density_file', type=util.parse_file, default=None,
                        help='CSV file containing max cut density for each problem.')
    util.add_boolean_flag(
        parser, 'verbose', default=False, help='Extra output.')
    args = parser.parse_args()
    create_dataset(args)
