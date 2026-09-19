"""Compatibility shim for the Python 3.8 host's pre-PEP-660 pip."""

from setuptools import find_packages, setup


setup(
    name="streamtimelens",
    version="0.1.0",
    description="Auditable delayed-query streaming video temporal grounding",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=5.3",
        "msgpack>=1.0",
        "numpy>=1.20",
        "Pillow>=7",
        "opencv-python-headless>=4.5",
        "transformers>=4.37,<5",
    ],
    extras_require={"dev": ["pytest>=7", "ruff>=0.1"], "jq01": ["torch>=2.0"]},
    entry_points={
        "console_scripts": [
            "streamtimelens-build-plan=streamtimelens.cli:build_arrival_plan_main",
            "streamtimelens-build-snapshots=streamtimelens.cli:build_snapshots_main",
            "streamtimelens-answer=streamtimelens.cli:answer_snapshots_main",
            "streamtimelens-oracle=streamtimelens.cli:run_oracle_refiner_main",
            "streamtimelens-generate-writer-feasibility=streamtimelens.cli:generate_writer_feasibility_main",
        ]
    },
)
