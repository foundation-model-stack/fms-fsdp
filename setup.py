from setuptools import find_packages, setup

setup(
    name="fms_fsdp",
    version="0.0.1",
    author="Linsong Chu, Davis Wertheimer, Brian Vaughan, Andrea Frittoli, Joshua Rosenkranz, Antoni Viros i Martin, Raghu Kiran Ganti",
    author_email="lchu@us.ibm.com",
    description="Pretraining scripts using FSDP and IBM Foundation Model Stack",
    url="https://github.com/foundation-model-stack/fms-fsdp",
    packages=find_packages(),
    install_requires=["ibm-fms >= 0.0.3", "torch >= 2.1"],
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
