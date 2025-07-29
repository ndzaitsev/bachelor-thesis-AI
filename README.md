Bachelor’s Thesis
# Integrating LORD Rule Learner with Scikit-learn module by Python–Java Proxy Approach
### Author: Mykola Zaitsev
### Thesis Supervisor: Prof. Dr. Johannes Fürnkranz
### Assistant Thesis Supervisor: Dr. Van Quoc Phuong Huynh, MSc
-----
Materials:
- Proxy model (.py file)
- Code for experiments (Jupyter notebook)
-----
## Steps of usage
In order to use proxy model several preparatory steps need to be taken.
### Download LORD
Download the LORD Java package from the following link:
[https://github.com/vqphuynh/LORD](https://github.com/vqphuynh/LORD)

### Compilation
Compile jar file with following commands
```console
javac -d bin -cp libs/* $(find src -name "*.java")
jar cf lord.jar -C bin 
```
Check that compilation was successful
```console
jar tf lord.jar 
```

### Internal configuration
In proxy model Python file specify the path to the .jar file
```python
jar_path = "/LORD-master/lord.jar"
```
