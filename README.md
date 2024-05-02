# XZZXdY
The repository pertaining to the code used in "Quantum error correcting Clifford deformed surface codes". It uses the Python packages qecsim and qsdxzzx and extends on these when needed 
## Repository structure
### XZZXdY code
This extends qecsim and qsdxzzx and is contained in all _rotatedplanarxzzxdy files. It changes the plaquette structure used in the XZZX code from qsdxzzx to get the XZZXdY structure. Changes are made to the rmps decoder from qecsim to use it on XZZXdY.
### XY code
This extends only qecsim and is contained in all _rotatedplanarxy files. It changes the plaquette structure used in the XZ code from qecsim to get the XY structure. Some changes are made to the rmps decoder from qecsim to apply it on XY.
### Modified_app
Modified_app has the "run" functions used to generate all the required data to run the models
### Simulate
Simulate generates the results using functions from modified_app for the specified code distances and variables, it then saves that data to a .json file. 
