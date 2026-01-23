
from netsquid.components.models.qerrormodels import T1T2NoiseModel,DepolarNoiseModel,DephaseNoiseModel

import sys
scriptpath = "./"
sys.path.append(scriptpath)

from Simulator import run_BB84_sim, run_BB84_sim_with_noise

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
        
    mymemNoiseMmodel=T1T2NoiseModel(T1=10**6, T2=10**5)
    myprocessorNoiseModel=DephaseNoiseModel(dephase_rate=0.004,time_independent=True)

    toWrite=run_BB84_sim(runtimes=1,num_bits=100,fibreLen=5
        ,memNoiseMmodel=mymemNoiseMmodel,processorNoiseModel=myprocessorNoiseModel,fibreNoise=0 
        ,sourceFreq=12e4,lenLoss=0.045
        ,qSpeed=2.083*10**5,cSpeed=2.083*10**5) #10**-9  
    
    logger.info("BB84 summary: runs=%s", len(toWrite[2]))
    logger.info("BB84 key lengths A:%s", [len(k) for k in toWrite[0]])
    logger.debug("BB84 key list A:%s", toWrite[0])
    logger.debug("BB84 key list B:%s", toWrite[1])
    logger.debug("BB84 key rate list:%s", toWrite[2])

    keyrate=sum(toWrite[2])/len(toWrite[2])
    logger.info("BB84 average key rate:%s", keyrate)
    logger.info("BB84 cost/bit/sec:%s", 4 / keyrate)

    print()

    noisy = run_BB84_sim_with_noise(
        runtimes=1,
        num_bits=100,
        fibreLen=5,
        memNoiseMmodel=mymemNoiseMmodel,
        processorNoiseModel=myprocessorNoiseModel,
        fibreNoise=0,
        sourceFreq=12e4,
        lenLoss=0.045,
        qSpeed=2.083 * 10**5,
        cSpeed=2.083 * 10**5,
        bit_flip_prob=0.02,
        cascade_block_size=16,
        cascade_rounds=4,
    )
    logger.info("BB84+noise summary: runs=%s", len(noisy[3]))
    logger.info("BB84+noise key lengths A:%s", [len(k) for k in noisy[0]])
    logger.info("BB84+noise key lengths B (after):%s", [len(k) for k in noisy[2]])
    logger.debug("BB84+noise key list A:%s", noisy[0])
    logger.debug("BB84+noise key list B (before):%s", noisy[1])
    logger.debug("BB84+noise key list B (after):%s", noisy[2])
    logger.debug("BB84+noise key rate list:%s", noisy[3])
