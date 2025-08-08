from setuptools import setup, find_packages
import textwrap

setup(
	name="jsASR",
	version="0.1.0",
	description="A lightweight ASR system to compute phoneme errors on a corpus such as Timit",
	author="Josef Schlittenlacher",
	maintainer="Marcus Ng",
	maintainer_email="marcusngzhijie@gmail.com",
	long_description=textwrap.dedent("""
		This package was originally authored by Josef Schlittenlacher.
		It has been updated and is currently maintained by Marcus Ng.
	"""),
    long_description_content_type="text/plain",
	packages=find_packages(),
	install_requires = [
		# TensorFlow per-OS
		'tensorflow==2.10.*; platform_system == "Windows" and python_version < "3.11"',
		'tensorflow[and-cuda]==2.17.*; platform_system == "Linux"',

		# ===== Windows (TF 2.10) compatible pins =====
		'numpy>=1.21,<1.24; platform_system == "Windows"',
		'scipy>=1.7,<1.10; platform_system == "Windows"',
		'h5py<3.8; platform_system == "Windows"',
		'scikit-learn>=1.1,<1.3; platform_system == "Windows"',

		# ===== Linux (TF 2.17) compatible pins =====
		# TF 2.17 wheels expect modern NumPy/SciPy
		'numpy>=1.26,<2.0; platform_system == "Linux"',
		'scipy>=1.10,<1.14; platform_system == "Linux"',
		'h5py>=3.10; platform_system == "Linux"',
		'scikit-learn>=1.5,<1.6; platform_system == "Linux"',
		
		# ===== Common (both OS) =====
		'python-dotenv>=1.1.1',
		'pandas>=2.3.1,<3.0.0',
		'seaborn>=0.13.2,<0.14.0',
		'bidict>=0.23.1',
		'pyprojroot>=0.3.0',
        'tqdm>=4.66.0'
	],
	python_requires=">=3.10,<3.12"
)