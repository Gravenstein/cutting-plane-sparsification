#!/usr/bin/python3

import io
import sys
import subprocess
import shutil
import os
import logging
from datetime import datetime
import argparse
from typing import List, Tuple
from tqdm import tqdm

# autopep8: off
# SLURM copies the script to a local directory, so we must make sure we can find local modules.
sys.path.append(os.getcwd())
import util
# autopep8: on

# Global logging configuration.
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(util.LOG_FORMATTER)
consoleHandler.setLevel(logging.INFO)
LOGGER.addHandler(consoleHandler)

_MEMORY_LIMIT_MB = 92000


def format_log_file(result_dir, task_id, seed, extension: str = 'out') -> str:
    filename = f'log-{task_id}_{seed}'
    if extension:
        filename = f'{filename}.{extension}'
    return os.path.join(result_dir, filename)


def append_log(file, log_file_path: str, header: str) -> None:
    file.write(f'@01 {header}\n')

    if not os.path.isfile(log_file_path):
        LOGGER.info(f'No file {log_file_path}, appending blank line instead.')
        file.write('\n')
        return

    with open(log_file_path, 'r') as f:
        for line in f:
            file.write(line)


def run_command(cmd: List[str], log_output=False, **subprocess_args) -> None:
    cmd = list(map(str, cmd))

    if log_output:
        assert not 'stdout' in subprocess_args, 'Cannot further redirect stdout'
        assert not 'stderr' in subprocess_args, 'Cannot further redirect stderr'
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **subprocess_args)
        with process.stdout as pipe:
            for line in iter(pipe.readline, b''):
                logging.info(line)
    else:
        process = subprocess.Popen(cmd, **subprocess_args)

    retcode = process.wait()
    sys.stdout.flush()
    if retcode != 0:
        LOGGER.error(
            f'Subprocess exited with status {retcode} != 0.')
        LOGGER.error('The command was:')
        for i, arg in enumerate(cmd):
            LOGGER.error(f'argv[{i}] = {arg}')
        LOGGER.error('Command in easy copy format: ' + " ".join(cmd))
        raise RuntimeError(f'{" ".join(cmd)} return {retcode} != 0')


def generate_contiguous_array(array: List[int]) -> str:
    assert array, f'Array must contain at least one element'

    array = sorted(array)
    next_diff = [array[i+1] - array[i] for i in range(len(array)-1)]
    next_diff.append(0)

    ranges = []
    at = 0
    while at < len(array):
        start = array[at]
        while next_diff[at] == 1:
            at += 1
        end = array[at]
        ranges.append((start, end))
        at += 1

    def format_range(r):
        if r[0] == r[1]:
            return f'{r[0]}'
        else:
            return f'{r[0]}-{r[1]}'

    return ','.join(map(format_range, ranges))


class Experiment:
    def __init__(self, test_file: str, model_directory: str, solution_directory: str) -> None:
        with open(test_file, 'r') as f:
            self.names = []
            self.model_files = []
            self.solution_files = []
            for line in f:
                name = line.strip()
                self.names.append(name)
                self.model_files.append(os.path.join(
                    model_directory, f'{name}.mps.gz'))
                self.solution_files.append(os.path.join(
                    solution_directory, f'{name}.sol.gz'))

    def get_task_settings(self, task_id: int) -> Tuple[str, str, str]:
        assert task_id < self.get_num_tasks(
        ), f'{task_id} exceeds number of tasks {self.get_num_tasks()}'

        util.assert_file_exists(self.model_files[task_id])
        util.assert_file_exists(self.solution_files[task_id])
        return (self.names[task_id],
                self.model_files[task_id],
                self.solution_files[task_id])

    def get_num_tasks(self) -> int:
        return len(self.names)


class Runner:
    def call(self, cmd_line: List[str], batch: bool, workers: List[int] = [], flags={}, **sub_process_args) -> None:
        raise NotImplementedError()


class SlurmRunner(Runner):
    def call(self, cmd_line: List[str], batch: bool, workers: List[int] = [], flags={}, **sub_process_args) -> None:
        slurm_command = ['sbatch' if batch else 'srun']
        if batch and workers:
            assert '--array' not in flags, 'Job array must be specified using `workers`, not `flags`'
            array = generate_contiguous_array(workers)
            flags['--array'] = array

        for key, value in flags.items():
            slurm_command.append(f'{key}')
            if value:
                slurm_command.append(f'{value}')
        if batch:
            run_command(slurm_command + cmd_line, **sub_process_args)
        else:
            if not workers:
                run_command(slurm_command + cmd_line)
            for i in workers:
                worker_command = [e for e in slurm_command]
                worker_command[f'--export=SLURM_ARRAY_TASK_ID={i}']
                LOGGER.info(f'Running worker {i}')
                run_command(worker_command + cmd_line, **sub_process_args)


class LocalRunner(Runner):
    def call(self, cmd_line: List[str], batch: bool, workers: List[int] = [], flags={}, **sub_process_args) -> None:
        if flags:
            LOGGER.debug(f'Running locally ignores flags: {flags}')
        if batch:
            LOGGER.debug(
                'Running locally does not support batch. Workers will be run sequentially.')
        if not workers:
            run_command(cmd_line, **sub_process_args)
        else:
            for i in workers:
                worker_env = os.environ.copy()
                worker_env['SLURM_ARRAY_TASK_ID'] = f'{i}'
                LOGGER.info(f'Running worker {i}...')
                run_command(cmd_line, env=worker_env, **sub_process_args)
                LOGGER.info(f'Worker {i} terminated normally')


def configure_and_compile(runner: Runner, source_path: str, build_path: str, debug_build: bool, n_threads=16) -> str:
    util.assert_dir_exists(source_path)
    os.makedirs(build_path, exist_ok=True)

    build_type = 'Debug' if debug_build else 'Release'
    configure_command = ['cmake', '-B', build_path, '-S', source_path,
                         f'-DCMAKE_BUILD_TYPE={build_type}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=1']
    build_command = ['cmake', '--build', build_path,
                     '--', '-j', f'{n_threads}']
    LOGGER.info(f'Configure: {build_path}')
    LocalRunner().call(configure_command, batch=False, log_output=True)
    LOGGER.info(f'Compile  : {build_path}')
    runner.call(build_command, batch=False, flags={
                '--cpus-per-task': 16, '--mem': _MEMORY_LIMIT_MB, '--time': '1:00:00', '--partition': 'REDACTED'}, log_output=True)
    return build_path


def submit(args):
    # We save log messages to a string so that we can dump it to the submit log later.
    log_string_stream = io.StringIO()
    string_log_handler = logging.StreamHandler(log_string_stream)
    string_log_handler.setFormatter(util.LOG_FORMATTER)
    string_log_handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(string_log_handler)

    LOGGER.debug(f'Command line: {" ".join(sys.argv)}')
    LOGGER.debug(f'Args: {args}')

    if args.gdb:
        assert args.mode == 'debug', 'You must compile in debug mode to run in GDB.'
        assert not args.run_remotely, 'Using GDB requires a local run.'

    experiment = Experiment(args.test_file, args.model_dir, args.solution_dir)

    # Setup basic parameters
    curdir = os.path.abspath(os.curdir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    benchmark_name = f'{args.name}-{timestamp}'
    workers = args.worker if args.worker else range(experiment.get_num_tasks())

    # Verify that all required files exist.
    for worker in workers:
        experiment.get_task_settings(worker)

    if args.min_density > 0:
        max_densities = util.read_csv_column(args.density_file, 'maxdensity')

        def density_filter(worker: int) -> bool:
            name, _, _ = experiment.get_task_settings(worker)
            return float(max_densities[name]) >= args.min_density
        workers = list(filter(density_filter, workers))

    # Verify that all problems exists in the nnz file, if provided.
    if args.nnz_file:
        nnz_data = util.read_csv_column(args.nnz_file, 'nnz')
        for worker in workers:
            name, _, _ = experiment.get_task_settings(worker)
            assert name in nnz_data, f'No value for {name} in {args.nnz_file}.'
            assert int(nnz_data[name]) >= - \
                1, f'Value for {name} in {args.nnz_file} is {nnz_data[name]} but must be non-negative or -1.'

    # Setup runner
    if args.run_remotely:
        runner = SlurmRunner()
    else:
        runner = LocalRunner()

    # Compile executable
    build_dir = os.path.join(args.build_dir, args.mode)
    executable = os.path.join(build_dir, 'sparsification')
    debug_build = args.mode == 'debug'
    configure_and_compile(runner, curdir, build_dir,
                          debug_build, n_threads=args.compilation_threads)
    util.assert_file_exists(executable)

    exec_args = []
    if args.gdb:
        exec_args = ['--args', executable] + exec_args
        executable = '/usr/bin/gdb'
    if args.dry_run:
        exec_args = [executable] + exec_args
        executable = '/usr/bin/echo'

    # Setup result directory
    LOGGER.info('Setup output directory')
    result_dir = os.path.join(args.result_dir, benchmark_name)
    assert not os.path.exists(result_dir), f'{result_dir} already exists'
    os.mkdir(result_dir)
    LOGGER.info('Copy config file')
    shutil.copyfile(args.config, os.path.join(
        result_dir, os.path.basename(args.config)))
    LOGGER.info('Copy CSV with problems used')
    shutil.copyfile(args.test_file, os.path.join(
        result_dir, os.path.basename(args.test_file)))
    if args.nnz_file:
        shutil.copyfile(args.nnz_file, os.path.join(
            result_dir, os.path.basename(args.nnz_file)))

    # Submit one job for each seed. This means we can run up to 1000 instances without
    # exceeding the max SLURM array size.
    for seed in args.seed:
        sbatch_args = {'--output': format_log_file(result_dir, '%a', seed),
                       '--error': format_log_file(result_dir, '%a', seed, extension='err'),
                       '--job-name': f'{benchmark_name}_{seed}',
                       '--mem': 32000,
                       '--cpus-per-task': 1,
                       '--time': '2:00:00'}
        if args.mode == 'performance':
            sbatch_args['--exclusive'] = ''
            sbatch_args['--partition'] = 'REDACTED'
            sbatch_args['--constraint'] = 'REDACTED'
        else:
            assert args.mode == 'debug', f'Unknown mode: {args.mode}.'
            assert '--exclusive' not in sbatch_args, f'Debug should not run exclusively.'
            sbatch_args['--partition'] = 'REDACTED'
            sbatch_args['--constraint'] = 'REDACTED'
        if args.nice:
            # This argument must be written FLAG=ARG, not FLAG ARG. Otherwise sbatch interprets
            # the niceness as the script name for some reason.
            sbatch_args['--nice=20000'] = ''
        if args.slurm_flag:
            for arg in args.slurm_flag:
                sbatch_args[arg] = ''

        worker_cmd = [__file__, 'run']
        worker_args = ['--test_file', args.test_file,
                       '--model_dir', args.model_dir,
                       '--solution_dir', args.solution_dir,
                       '--result_dir', result_dir,
                       '--executable', executable,
                       '--config', args.config]
        if args.default_config:
            worker_args.append('--default_config')
        else:
            worker_args.append('--no-default_config')
        for arg in exec_args:
            worker_args.append('--arg')
            worker_args.append(f'"{arg}"')
        worker_args.append('--seed')
        worker_args.append(seed)
        if args.nnz_file:
            worker_args.append('--nnz_file')
            worker_args.append(args.nnz_file)

        LOGGER.info('Running workers')
        runner.call(worker_cmd + worker_args, batch=True, workers=workers,
                    flags=sbatch_args, log_output=args.run_remotely)
    if (args.run_remotely):
        LOGGER.info('Your SLURM queue:')
        LocalRunner().call(['squeue', '--user=REDACTED'],
                           batch=False, log_output=True)

    submit_log_file = os.path.join(result_dir, 'submit_log.txt')
    LOGGER.info(
        f'Copying log from `submit` to {submit_log_file}. Messages after this point will not be included.')
    with open(submit_log_file, 'w') as log_file:
        log_file.write(log_string_stream.getvalue())
    LOGGER.removeHandler(string_log_handler)


def run(args):
    experiment = Experiment(args.test_file, args.model_dir, args.solution_dir)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    name, problem_file, solution_file = experiment.get_task_settings(
        task_id)
    result_dir = os.path.join(args.result_dir, f'{name}_{args.seed}')
    assert not os.path.exists(result_dir), f'{result_dir} already exists'
    os.mkdir(result_dir)
    if args.nnz_file:
        nnz_limit = util.read_csv_column(args.nnz_file, 'nnz')[name]
    else:
        nnz_limit = -1
    sparsification_arguments = ['--problem_name', name,
                                '--config_file', args.config,
                                '--problem_file', problem_file,
                                '--solution_file', solution_file,
                                '--result_dir', result_dir,
                                '--seed', args.seed,
                                '--max_selected_nnz', nnz_limit,
                                '--default_config' if args.default_config else '--nodefault_config']
    exec_args = [] if not args.arg else [arg.strip('"\'') for arg in args.arg]
    LocalRunner().call([args.executable] + exec_args +
                       sparsification_arguments, batch=False)
    LOGGER.info('Run finished normally.')


def finalize(args):
    experiment = Experiment(args.test_file, args.model_dir, args.solution_dir)
    result_dir = args.result_dir[0]

    with open(os.path.join(result_dir, 'all.out'), 'w') as all_logs_file:
        for seed in args.seed:
            with open(os.path.join(result_dir, f'{seed}.out'), 'w') as seed_logs_file:
                for task_id in tqdm(range(experiment.get_num_tasks()), desc=f'Appending logs for seed {seed}'):
                    log_file = util.find_file(format_log_file(
                        '', task_id, seed, ''), args.result_dir, ['out', 'txt'])
                    util.assert_file_exists(log_file)
                    name, _, _ = experiment.get_task_settings(task_id)
                    append_log(all_logs_file, log_file, f'{name}_{seed}')
                    append_log(seed_logs_file, log_file, name)

    with open(os.path.join(result_dir, 'all.err'), 'w') as all_err_file:
        for seed in args.seed:
            with open(os.path.join(result_dir, f'{seed}.err'), 'w') as seed_err_file:
                for task_id in tqdm(range(experiment.get_num_tasks()), desc=f'Appending error logs for seed {seed}'):
                    err_file = util.find_file(format_log_file(
                        '', task_id, seed, ''), args.result_dir, ['err'])
                    name, _, _ = experiment.get_task_settings(task_id)
                    append_log(all_err_file, err_file, f'{name}_{seed}')
                    append_log(seed_err_file, err_file, name)

    testset = os.path.basename(args.test_file).split('.')[0]
    queue = 'REDACTED' if args.mode == 'debug' else 'REDACTED'
    for seed in args.seed:
        with open(os.path.join(result_dir, f'{seed}.meta'), 'w') as meta_file:
            meta_file.write(f'@Seed {seed}\n')
            meta_file.write(f'@Settings {args.name}\n')
            meta_file.write(f'@TstName {testset}\n')
            meta_file.write('@BinName sparsification\n')
            meta_file.write(f'@MemLimit {_MEMORY_LIMIT_MB}\n')
            meta_file.write(f'@FeasTol {1e-6}\n')
            meta_file.write(f'@Queue {queue}\n')
            if args.mode == 'performance':
                meta_file.write(f'@Exclusive\n')


if __name__ == '__main__':
    LOGGER.info('Called script: ' + ' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Parameters for runner
    submit_parser = subparsers.add_parser(
        'submit', help='Configure and submit jobs to SLURM.')
    submit_parser.set_defaults(func=submit)
    submit_parser.add_argument('--test_file', type=util.parse_file, required=True,
                               help='File containing problem names, one per line.')
    submit_parser.add_argument('--model_dir', type=util.parse_dir,
                               help='Directory where the model files are located.')
    submit_parser.add_argument('--solution_dir', type=util.parse_dir,
                               help='Directory where the solution files are located.')
    submit_parser.add_argument(
        '--name', type=str, required=True, help='Experiment name.')
    submit_parser.add_argument('-s', '--seed', type=int, action='append', required=True,
                               help='SCIP seeds, repeatable.')
    submit_parser.add_argument(
        '--config', type=util.parse_file, required=True, help='SCIP config file.')
    submit_parser.add_argument('--result_dir', type=util.parse_dir, required=True,
                               help='Directory where results should be placed. Interpretation depends on subcommand.')
    submit_parser.add_argument('--nnz_file', type=util.parse_file,
                               required=False, help='CSV file containing cut nnz limit per problem.')
    submit_parser.add_argument(
        '--mode', type=str, required=True, choices=['performance', 'debug'])
    util.add_boolean_flag(submit_parser, 'dry_run', default=False,
                          help='If enabled, the executable is not actually run. Useful to test correct configuration.')
    util.add_boolean_flag(submit_parser, 'nice', default=False,
                          help='Allow other users in the same group to skip ahead in line.')
    util.add_boolean_flag(submit_parser, 'run_remotely', default=True,
                          help='Run the jobs remotely use slurm.')
    submit_parser.add_argument('--worker', type=int, action='append',
                               required=False, help='Only run these worker indices.')
    submit_parser.add_argument('--compilation_threads', type=int, default=16,
                               help='Number of threads to use for compilation.')
    util.add_boolean_flag(submit_parser, 'gdb', default=False,
                          help='Run the executable in GDB. Requires debug and local run.')
    submit_parser.add_argument('--build_dir', type=util.parse_dir, default='build',
                               help='Path to where the executable should be built.')
    submit_parser.add_argument('--slurm_flag', type=str, action='append',
                               required=False, help='Additional flags to pass to the SLURM batch command.')
    util.add_boolean_flag(submit_parser, 'default_config',
                          default=False, help='Disable all custom SCIP plugins.')
    submit_parser.add_argument('--min_density', type=float, default=0.0,
                               help='Only run instances where at least one cut had this density.')
    submit_parser.add_argument('--density_file', type=util.parse_file, default='data/maxdensity.csv',
                               help='CSV file listing the maximum density of a cut generated for this instance.')

    # Parameters for worker
    run_parser = subparsers.add_parser(
        'run', help='Run a single job. Should only be called by the run mode.')
    run_parser.set_defaults(func=run)
    run_parser.add_argument('--test_file', type=util.parse_file, required=True,
                            help='File containing problem names, one per line.')
    run_parser.add_argument('--model_dir', type=util.parse_dir, default='REDACTED',
                            help='Directory where the model files are located.')
    run_parser.add_argument('--solution_dir', type=util.parse_dir, default='REDACTED',
                            help='Directory where the solution files are located.')
    run_parser.add_argument('-s', '--seed', type=int,
                            required=True, help='SCIP seed.')
    run_parser.add_argument('--result_dir', type=util.parse_dir, required=True,
                            help='Directory where results should be placed. Interpretation depends on subcommand.')
    run_parser.add_argument(
        '--config', type=util.parse_file, required=True, help='SCIP config file.')
    run_parser.add_argument('--executable', type=util.parse_file,
                            required=True, help='Experiment executable')
    run_parser.add_argument('--arg', type=str, action='append',
                            help='Arguments passed immediately after the executable, but before the script defined flags.')
    run_parser.add_argument('--nnz_file', type=util.parse_file,
                            required=False, help='CSV file containing cut nnz limit per problem.')
    util.add_boolean_flag(run_parser, 'default_config',
                          default=False, help='Disable all custom SCIP plugins.')

    # Parameters for finalize
    finalize_parser = subparsers.add_parser(
        'finalize', help='Finalizes results.')
    finalize_parser.set_defaults(func=finalize)
    finalize_parser.add_argument(
        '--name', type=str, required=True, help='Experiment name.')
    finalize_parser.add_argument('--result_dir', type=util.parse_dir, action='append', required=True,
                                 help='Directory where results should be placed. If repeated, each directory is checked until each log file is found.')
    finalize_parser.add_argument('--test_file', type=util.parse_file, required=True,
                                 help='File containing problem names, one per line.')
    finalize_parser.add_argument('--model_dir', type=util.parse_dir, default='REDACTED',
                                 help='Directory where the model files are located.')
    finalize_parser.add_argument('--solution_dir', type=util.parse_dir, default='REDACTED',
                                 help='Directory where the solution files are located.')
    finalize_parser.add_argument('-s', '--seed', type=int, action='append', required=True,
                                 help='SCIP seeds, repeatable.')
    finalize_parser.add_argument(
        '--mode', type=str, required=True, choices=['performance', 'debug'])

    # Run appropriate mode
    args = parser.parse_args()
    args.func(args)
