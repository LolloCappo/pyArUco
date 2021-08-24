Digital lock-in analysis
---------------------------------------------

.. image:: https://zenodo.org/badge/220818285.svg
   :target: https://zenodo.org/badge/latestdoi/220818285

Perform digital lock-in analysis built for thermal analysis


Installing this package
-----------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install pyLIA


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

	import pyLIA
	import numpy as np
	import matplotlib.pyplot as plt

	data = np.load('camera.npy') ## Thermal acquisition
	fs = 400  ## Sampling freqency of the thermal video [Hz]
	fl = 55  ## Load freqency of the excitation test [Hz]

	mag, ph = pyLIA.LIA(data, fs, fl)

	plt.figure()
	plt.imshow(mag)
	cbar = plt.colorbar()
	cbar.set_label('[unit]')

	plt.figure()
	plt.imshow(ph)
	cbar = plt.colorbar()
	cbar.set_label('[deg]')
    


Reference:
<https://www.sciencedirect.com/science/article/pii/S0142112320301924>
