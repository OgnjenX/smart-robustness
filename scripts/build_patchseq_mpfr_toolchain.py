"""Isolated pinned dependency build only; no neuron model, fitting or data access."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
from functools import partial
from pathlib import Path, PurePosixPath

import yaml

REG = Path('docs/validation-results/post2008-patchseq-mpfr-hypothesis-1503.yaml')
CFLAGS = '-O2 -fPIC -fno-fast-math -ffp-contract=off'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(directory):
    rows = []
    for path in sorted(directory.rglob('*')):
        name = str(path.relative_to(directory))
        if path.is_symlink():
            rows.append([name, 'link', os.readlink(path)])
        elif path.is_file():
            rows.append([name, 'file', path.stat().st_mode & 0o777, sha(path)])
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def unpack_checked(archive, destination, package):
    """A verified source archive may write only into its fresh dedicated subtree."""
    if destination.exists():
        raise ValueError('existing extraction subtree retained; refusing overwrite')
    if not hasattr(tarfile, 'data_filter'):
        raise RuntimeError('safe tar data filter is required; no unsafe fallback')
    with tarfile.open(archive, mode='r:xz') as tar:
        members = tar.getmembers()
        if not members or len(members) > 20000 or sum(m.size for m in members) > 100_000_000:
            raise ValueError('source archive extraction budget exceeded')
        for item in members:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != package:
                raise ValueError('archive member outside fixed source package')
        destination.mkdir(parents=True)
        tar.extractall(destination, members=members, filter='data')


def main():
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['toolchain_output']).resolve()
    expected = (Path.cwd()/'results'/'patchseq-mpfr-toolchain-1504').resolve()
    if root != expected:
        raise ValueError('dependency build may write only its fixed project-results directory')
    compiler = shutil.which('clang')
    make, curl = shutil.which('make'), shutil.which('curl')
    if not compiler or not make or not curl:
        raise RuntimeError('clang, make and curl are required')
    compiler = str(Path(compiler).resolve())
    root.mkdir(parents=True, exist_ok=True)
    prefix = root/'prefix'
    pins = {str(REG): sha(REG), str(Path(__file__)): sha(__file__), compiler: sha(compiler),
            str(Path(make).resolve()): sha(Path(make).resolve()), str(Path(curl).resolve()): sha(Path(curl).resolve())}
    configuration = {'source_hashes': pins, 'platform': platform.platform(), 'machine': platform.machine(),
                     'compiler_version': subprocess.run([compiler, '--version'], check=True,
                         capture_output=True, text=True, timeout=10).stdout,
                     'cflags': CFLAGS, 'prefix': str(prefix), 'jobs': 2,
                     'dependencies': reg['dependencies']}
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'CC': compiler,
                   'CFLAGS': CFLAGS, 'LC_ALL': 'C', 'MAKEINFO': 'true'}
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'checkpoint.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'configuration': configuration, 'stages': {}, 'source_trees': {}, 'products': {}}
        if state['configuration'] != configuration:
            raise ValueError('dependency checkpoint source/configuration differs')

        def retain():
            if any(sha(p) != h for p, h in pins.items()):
                raise ValueError('builder or toolchain input changed')
            temporary = path.with_suffix('.pending')
            with temporary.open('w') as file:
                json.dump(state, file, sort_keys=True, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)

        def stage(name, command=None, cwd=None, action=None):
            log = root/(name+'.log')
            if name in state['stages']:
                old = state['stages'][name]
                if old['status'] != 'passed' or old['log_sha256'] != sha(log):
                    raise ValueError('failed or changed dependency stage retained; no automatic replacement')
                return False
            if log.exists():
                raise ValueError('orphan dependency log retained; inspect before any restart')
            try:
                with log.open('xb') as file:
                    if command:
                        subprocess.run(command, cwd=cwd, env=environment, stdout=file,
                                       stderr=subprocess.STDOUT, check=True, timeout=1800)
                    else:
                        action()
                state['stages'][name] = {'status': 'passed', 'command': command,
                                         'log_sha256': sha(log)}
            except Exception as exc:
                state['stages'][name] = {'status': 'failed-retained-no-replacement',
                    'command': command, 'error': str(exc), 'log_sha256': sha(log)}
                state['status'] = 'dependency-build-failed-not-a-scientific-result'
                retain()
                raise
            retain()
            print('Dependency stage complete:', name, flush=True)
            return True

        for dependency in reg['dependencies']:
            name = dependency['name']
            archive = root/dependency['archive']
            if not dependency['url'].startswith(('https://ftp.gnu.org/gnu/', 'https://www.mpfr.org/')):
                raise ValueError('dependency source URL outside fixed official origins')
            if archive.exists() and sha(archive) != dependency['sha256']:
                raise ValueError('existing source archive hash differs; refusing overwrite')
            if not archive.exists():
                stage(name+'-download', [curl, '--fail', '--proto', '=https', '--tlsv1.2',
                    '--max-time', '120', '--max-filesize', '8000000', '--output', str(archive), dependency['url']])
            if sha(archive) != dependency['sha256']:
                raise ValueError('downloaded archive does not match frozen SHA256')
            source_root = root/(name+'-source')
            extracted = stage(name+'-extract', action=partial(unpack_checked, archive, source_root,
                                                             dependency['source_directory']))
            source = source_root/dependency['source_directory']
            if extracted:
                state['source_trees'][name] = tree_sha(source)
                retain()
            if state['source_trees'].get(name) != tree_sha(source):
                raise ValueError('extracted dependency source changed or extraction hash missing')
            if any(sha(p) != h for p, h in state['products'].items()):
                raise ValueError('previously installed dependency product changed')
            build = root/(name+'-build')
            build.mkdir(exist_ok=True)
            configure = [str(source/'configure'), '--prefix='+str(prefix), '--disable-shared', '--enable-static', '--with-pic']
            if name == 'mpfr':
                configure += ['--with-gmp='+str(prefix), '--enable-thread-safe']
            stage(name+'-configure', configure, build)
            stage(name+'-build', [make, '-j2', 'MAKEINFO=true'], build)
            stage(name+'-check', [make, '-j2', 'check', 'MAKEINFO=true'], build)
            stage(name+'-install', [make, 'install', 'MAKEINFO=true'], build)
            if tree_sha(source) != state['source_trees'][name]:
                raise ValueError('source tree changed during dependency compilation/tests')
            for relative in ('include/'+name+'.h', 'lib/lib'+name+'.a'):
                product = str(prefix/relative)
                current = sha(product)
                if product in state['products'] and state['products'][product] != current:
                    raise ValueError('installed dependency product changed')
                state['products'][product] = current
            retain()
        products = ['include/gmp.h', 'include/mpfr.h', 'lib/libgmp.a', 'lib/libmpfr.a']
        installed = {str(prefix/p): sha(prefix/p) for p in products}
        state.update(status='isolated-dependencies-built-and-upstream-tests-passed-not-backend-equivalence',
                     installed_sha256=installed, scientific_goal_complete=False,
                     fitting_restart_authorized=False, held_out_response_access=False)
        retain()
        print('Pinned isolated dependency toolchain complete; no neuron benchmark executed', flush=True)


if __name__ == '__main__':
    main()
