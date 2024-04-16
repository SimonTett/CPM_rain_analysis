# dodo script -- run with doit
from doit.tools import run_once
import pathlib
import os
figure_dir = pathlib.Path('figures')
figures = [
    {'script': 'plot_scatter.py', 'figure': figure_dir/'scatter.png'},
    {'script': 'plot_radar_jja.py', 'figure': figure_dir/'radar_jja.png'},
    # Add more dictionaries as needed
]

def task_create_figures():
    for figure in figures:
        def run_script():
            script = figure['script']
            print("Running ",script)
            os.system(f"python {script}")

        yield {
            'name': figure['figure'],
            'actions': [run_script],
            'file_dep': [figure['script']],
            'targets': [figure['figure']],
            'uptodate': [run_once]
        }