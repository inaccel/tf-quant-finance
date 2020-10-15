"""Setup inaccel package."""

import requests

from packaging import version as v
from setuptools import find_namespace_packages, setup

def bump(package, version):
    version = v.parse(version)
    try:
        pypi = requests.get('https://pypi.org/pypi/inaccel-' + package + '/json').json()
        releases = [v.parse(release) for release in pypi['releases']]
        releases = filter(lambda release: release.release[:-1] == version.release, releases)
        inaccel = tuple([max(releases).release[-1] + 1])
    except:
        inaccel = tuple([0])
    return '.'.join(str(x) for x in version.release + inaccel)

coral_api = '==1.8'

package = 'tf-quant-finance'
version = '0.0.1.dev24'

setup(
    name = 'inaccel-' + package,
    packages = find_namespace_packages(include = ['inaccel.*']),
    namespace_packages = ['inaccel'],
    version = bump(package, version),
    license = 'Apache-2.0',
    description = 'InAccel ' + package + '-like package',
    author = 'InAccel',
    author_email = 'info@inaccel.com',
    url = 'https://docs.inaccel.com',
    keywords = ['InAccel Coral', 'FPGA', 'inaccel', package],
    install_requires = [
        'coral-api' + coral_api, package + '==' + version,
    ],
    include_package_data = True,
    zip_safe = True,
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires = '>=3.6',
)
