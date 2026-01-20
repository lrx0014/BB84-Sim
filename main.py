
from netsquid.components.models.qerrormodels import T1T2NoiseModel,DepolarNoiseModel,DephaseNoiseModel

import sys
scriptpath = "./"
sys.path.append(scriptpath)

from Simulator import run_BB84_sim

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
        
    mymemNoiseMmodel=T1T2NoiseModel(T1=10**6, T2=10**5)
    myprocessorNoiseModel=DephaseNoiseModel(dephase_rate=0.004,time_independent=True)

    toWrite=run_BB84_sim(runtimes=3,num_bits=100,fibreLen=5
        ,memNoiseMmodel=mymemNoiseMmodel,processorNoiseModel=myprocessorNoiseModel,fibreNoise=0 
        ,sourceFreq=12e4,lenLoss=0.045
        ,qSpeed=2.083*10**5,cSpeed=2.083*10**5) #10**-9  
    
    logger.info("key list A:%s", toWrite[0])
    logger.info("key list B:%s", toWrite[1])
    logger.info("key rate list:%s", toWrite[2])

    keyrate=sum(toWrite[2])/len(toWrite[2])
    logger.info("Average key rate:%s", keyrate)
    logger.info("cost/bit/sec :%s", 4 / keyrate)
