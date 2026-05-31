from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''Reads the requirements from the specified file and returns them as a list.'''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="student_mark_prediction_ml",
    version="0.1",
    packages=find_packages(),
    author ='Sital',
    author_email='sitaladhikari134@gmail.com',
    install_requires= get_requirements('requirements.txt')
)