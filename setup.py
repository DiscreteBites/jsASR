from setuptools import setup, find_packages

setup(
	name="jsASR",
	version="0.1.0",
	description="A lightweight ASR system to compute phoneme errors on a corpus such as Timit",
	author="Josef Schlittenlacher",
	maintainer="Marcus Ng",
	maintainer_email="marcusngzhijie@gmail.com",
	long_description="""
	This package was originally authored by Josef Schlittenlacher.
	It has been updated and is currently maintained by Marcus Ng.
	""",
    long_description_content_type="text/plain",
	packages=find_packages(),
	install_requires=[
		"tensorflow==2.10",        # Last version supporting non-Python 3.11+ issues
		"scikit-learn>=1.1,<1.3",  
		"numpy>=1.21,<1.24",       
		"scipy>=1.7,<1.10",        # Required by TF and sklearn
		"h5py<3.8",                # TensorFlow 2.10 requirement
		"python-dotenv>=1.1.1",
		"pandas>=2.3.1,<3.0.0",
		"seaborn>=0.13.2,<0.14.0",
		"bidict>=0.23.1",
		"pyprojroot>=0.3.0"
	],

	python_requires=">=3.10,<3.12"
)