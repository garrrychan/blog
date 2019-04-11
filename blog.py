import os
import subprocess
from fire import Fire

def shell(command):
    '''Execute bash commands'''
    if isinstance(command, str):
        command = [command]
    [subprocess.call([c], shell=True) for c in command]

def html(pelican='pelicanconf.py', output='output', content='content', theme='theme'):
    '''Build website artifacts'''
    shell(f'pelican -s {pelican} -o {output} -t {theme} {content}')

def local(output='output'):
    '''Preview website content'''
    os.environ['PELICAN_ENV'] = 'DEV'
    html(output=output)
    shell(f'cd {output}; python -m http.server')

def publish(output='output', branch='gh-pages'):
    '''Push content to GitHub Pages'''
    os.environ['PELICAN_ENV'] = 'PROD'
    html(output=output)
    shell([
        f'ghp-import -m "Generate Pelican site" -b {branch} {output}',
        f'git push origin {branch}'
    ])

def convert(notebook, input='jupyter', output='content'):
    '''Convert a jupyter notebook to a pelican-compatible markdown file'''
    shell([
        f'cp {input}/{notebook}.ipynb {output}/{notebook}.ipynb',
        f'cd {output}; jupyter nbconvert --to markdown {notebook}.ipynb',
        f'cd {output}; rm {notebook}.ipynb'
    ])
    if os.path.isdir(f'{output}/{notebook}_files'):
        shell([
            f'cd {output}; cp -a {notebook}_files/. images/',
            f'cd {output}; rm -rf {notebook}_files'
        ])
    with open(f'{output}/{notebook}.md', encoding='UTF-8') as f:
        chapter = f.read().strip()
        chapter = chapter.replace(f'({notebook}_files/', '(images/')
    with open(f'{output}/{notebook}.md', 'w', encoding='UTF-8') as f:
        f.write(chapter)

def flush(output='output'):
    shell(f'rm -rf {output}')

if __name__ == '__main__':
    Fire({
        'local': local,
        'publish': publish,
        'convert': convert,
        'flush': flush
    })
