__version__ = '0.6'
import numpy as np

def LIA(data,fs,fl):
    """
    Function to obtain magnitude and phase of a thermal acquisition usign a lock-in analyzer
    
    Arguments:
    ---------
        data : array
                Sequence of thermal images [frames, height, width]

        fs   : float
                Sampling frequency of thermal video [Hz]

        fl   : float
                Frequency of excitaiton load [Hz]
    
    Returns:
    -------
        magnitude : Magnitude of locked-in signal [units of thermal video]
        phase  : Phase of locked-in signal [deg -- degrees]

    """
    t = np.linspace(0, data.shape[0]/fs, data.shape[0], endpoint = True) # Time vector
    sine = np.sin(fl * t * 2 * np.pi) # Sine wave at the expected thermal frequency (i.e. the load frequency)
    cosine = np.cos(fl * t * 2 * np.pi) # Cosine wave at the expected thermal frequency (i.e. the load frequency)

    S = (np.ones((data.shape[0], data.shape[1], data.shape[2])).T*sine).T 
    C = (np.ones((data.shape[0], data.shape[1], data.shape[2])).T*cosine).T 

    L1 = S * data # sine matrix * data matrix
    L2 = C * data # cosine matrix * data matrix
    Re = 2 * np.trapz(L1, axis = 0)/data.shape[0] # real part
    Img = 2 * np.trapz(L2, axis = 0)/data.shape[0] #imaginary part
    
    magnitude = np.sqrt(Re**2 + Img**2) # magnitude
    phase = np.arctan(Img/Re) * 180/np.pi #phase in degrees

    return magnitude, phase
