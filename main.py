
from netsquid.components.models.qerrormodels import T1T2NoiseModel,DepolarNoiseModel,DephaseNoiseModel

import sys
scriptpath = "./"
sys.path.append(scriptpath)

from simulator import (
    run_BB84_sim,
    run_BB84_sim_with_noise,
    run_BB84_sim_with_noise_overflow,
)

import logging
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    mymemNoiseMmodel=T1T2NoiseModel(T1=10**6, T2=10**5)
    myprocessorNoiseModel=DephaseNoiseModel(dephase_rate=0.004,time_independent=True)

    toWrite=run_BB84_sim(runtimes=1,num_bits=100,fibreLen=5
        ,memNoiseMmodel=None,processorNoiseModel=None,fibreNoise=0 
        ,sourceFreq=12e4,lenLoss=0.045
        ,qSpeed=2.083*10**5,cSpeed=2.083*10**5) #10**-9  
    
    logger.info("BB84_baseline summary: runs=%s", len(toWrite[5]))
    logger.info("BB84_baseline key lengths A:%s", [len(k) for k in toWrite[0]])
    logger.info("BB84_baseline final key lengths A:%s", [len(k) for k in toWrite[3]])
    logger.info("BB84_baseline final key lengths B:%s", [len(k) for k in toWrite[4]])
    logger.debug("BB84_baseline key list A:%s", toWrite[0])
    logger.debug("BB84_baseline key list B:%s", toWrite[1])
    logger.debug("BB84_baseline final key list A:%s", toWrite[3])
    logger.debug("BB84_baseline final key list B:%s", toWrite[4])
    logger.debug("BB84_baseline key rate list:%s", toWrite[5])

    keyrate=sum(toWrite[5])/len(toWrite[5])
    logger.info("BB84_baseline average key rate:%s", keyrate)
    logger.info("BB84_baseline cost/bit/sec:%s", 4 / keyrate)

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
    logger.info("BB84+noise summary: runs=%s", len(noisy[5]))
    logger.info("BB84+noise key lengths A:%s", [len(k) for k in noisy[0]])
    logger.info("BB84+noise key lengths B (after):%s", [len(k) for k in noisy[2]])
    logger.info("BB84+noise final key lengths A:%s", [len(k) for k in noisy[3]])
    logger.info("BB84+noise final key lengths B:%s", [len(k) for k in noisy[4]])
    logger.debug("BB84+noise key list A:%s", noisy[0])
    logger.debug("BB84+noise key list B (before):%s", noisy[1])
    logger.debug("BB84+noise key list B (after):%s", noisy[2])
    logger.debug("BB84+noise final key list A:%s", noisy[3])
    logger.debug("BB84+noise final key list B:%s", noisy[4])
    logger.debug("BB84+noise key rate list:%s", noisy[5])

    print()

    overflow = run_BB84_sim_with_noise_overflow(
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
    logger.info("BB84+abort summary: runs=%s", len(overflow[5]))
    logger.info("BB84+abort key lengths A:%s", [len(k) for k in overflow[0]])
    logger.info("BB84+abort key lengths B (after):%s", [len(k) for k in overflow[2]])
    logger.info("BB84+abort final key lengths A:%s", [len(k) for k in overflow[3]])
    logger.info("BB84+abort final key lengths B:%s", [len(k) for k in overflow[4]])
    logger.debug("BB84+abort key list A:%s", overflow[0])
    logger.debug("BB84+abort key list B (before):%s", overflow[1])
    logger.debug("BB84+abort key list B (after):%s", overflow[2])
    logger.debug("BB84+abort final key list A:%s", overflow[3])
    logger.debug("BB84+abort final key list B:%s", overflow[4])
    logger.debug("BB84+abort key rate list:%s", overflow[5])
