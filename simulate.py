#imports
import modified_app
from qecsim.models.generic import BiasedDepolarizingErrorModel
from _rotatedplanarxzzxdycode import RotatedPlanarXZZXdYCode
from _rotatedplanarxzzxdyrmpsdecoder import RotatedPlanarXZZXdYRMPSDecoder
import logging
import numpy as np

logger = logging.getLogger(__name__)

#set parameters
eta = 300   #Z bias
min_error_prob, max_error_prob = 0, (1+1/eta)/(2+1/eta)
num_points = 25

min_code_distance = 9
max_code_distance = 51
interval = 6

max_runs = 1000
filename = 'eta_300_d9-51.json'
#run simulation
codes = [RotatedPlanarXZZXdYCode(distance) for distance in range(min_code_distance, max_code_distance+1, interval)]
error_probabilities = np.linspace(min_error_prob, max_error_prob, num_points)
decoder = RotatedPlanarXZZXdYRMPSDecoder(chi=12)
error_model = BiasedDepolarizingErrorModel(eta, axis='Z')

data = [modified_app.run(code, error_model, decoder, error_probability, max_runs) for code in codes for error_probability in error_probabilities]

modified_app.save_data(filename, data)
print('Completed')
