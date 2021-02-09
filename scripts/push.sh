rm dist/
python3.7 setup.py sdist bdist_wheel
python3.7 -m twine upload --repository testpypi dist/* 
